use std::convert::Infallible;
use std::sync::Arc;

use alloy_consensus::EMPTY_OMMER_ROOT_HASH;
use alloy_primitives::{Address, B256, U256};
use alloy_rlp::{Decodable, Encodable};
use alloy_rpc_types::{
    engine::{
        ExecutionPayloadEnvelopeV2, ExecutionPayloadEnvelopeV3, ExecutionPayloadEnvelopeV4,
        ExecutionPayloadV1, PayloadAttributes as EthPayloadAttributes, PayloadId,
    },
    Withdrawal,
};
use clap::Parser;
use jsonrpsee::{core::RpcResult, proc_macros::rpc};
use reth::{
    api::PayloadTypes,
    builder::{
        components::{ComponentsBuilder, EngineValidatorBuilder, PayloadServiceBuilder},
        node::{NodeTypes, NodeTypesWithEngine},
        BuilderContext, FullNodeTypes, Node, PayloadBuilderConfig,
    },
    chainspec::EthereumChainSpecParser,
    cli::Cli,
    providers::{CanonStateSubscriptions, StateProviderFactory},
    transaction_pool::TransactionPool,
};
use reth_basic_payload_builder::{
    commit_withdrawals, is_better_payload, BasicPayloadJobGenerator,
    BasicPayloadJobGeneratorConfig, BuildArguments, BuildOutcome, PayloadBuilder, PayloadConfig,
    WithdrawalsOutcome,
};
use reth_chain_state::ExecutedBlock;
use reth_chainspec::{ChainSpec, ChainSpecProvider, EthereumHardforks};
use reth_errors::RethError;
use reth_ethereum_payload_builder::EthereumPayloadBuilder;
use reth_evm::{system_calls::SystemCaller, ConfigureEvm, ConfigureEvmEnv, NextBlockEnvAttributes};
use reth_evm_ethereum::{eip6110::parse_deposits_from_receipts, EthEvmConfig};
use reth_execution_types::ExecutionOutcome;
use reth_node_api::{
    payload::{EngineApiMessageVersion, EngineObjectValidationError, PayloadOrAttributes},
    validate_version_specific_fields, EngineTypes, EngineValidator, PayloadAttributes,
    PayloadBuilderAttributes,
};
use reth_node_ethereum::node::{
    EthereumAddOns, EthereumConsensusBuilder, EthereumExecutorBuilder, EthereumNetworkBuilder,
    EthereumPoolBuilder,
};
use reth_payload_builder::{
    EthBuiltPayload, EthPayloadBuilderAttributes, PayloadBuilderError, PayloadBuilderHandle,
    PayloadBuilderService,
};
use reth_primitives::{
    constants::BEACON_NONCE,
    proofs::{self, calculate_requests_root},
    revm_primitives::{
        calc_excess_blob_gas, BlockEnv, CfgEnvWithHandlerCfg, EVMError, EnvWithHandlerCfg,
        InvalidTransaction, ResultAndState,
    },
    Block, BlockBody, Header, Receipt, TransactionSignedEcRecovered, Withdrawals,
};
use reth_revm::{
    database::StateProviderDatabase, db::states::bundle_state::BundleRetention, DatabaseCommit,
    State,
};
use reth_trie::HashedPostState;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

fn main() {
    Cli::<EthereumChainSpecParser, RethCliFocilExt>::parse()
        .run(|builder, args| async move {
            let handle = builder
                .node(FocilNode::default())
                .extend_rpc_modules(move |ctx| {
                    if !args.enable_ext {
                        return Ok(());
                    }

                    let pool = ctx.pool().clone();
                    let ext = FocilExt { pool };

                    ctx.modules.merge_configured(ext.into_rpc())?;

                    Ok(())
                })
                .launch()
                .await?;

            handle.wait_for_node_exit().await
        })
        .unwrap();
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FocilPayloadAttributes {
    #[serde(flatten)]
    pub inner: EthPayloadAttributes,
    pub inclusion_list: Vec<Vec<u8>>,
}

impl PayloadAttributes for FocilPayloadAttributes {
    fn timestamp(&self) -> u64 {
        self.inner.timestamp()
    }

    fn withdrawals(&self) -> Option<&Vec<Withdrawal>> {
        self.inner.withdrawals()
    }

    fn parent_beacon_block_root(&self) -> Option<B256> {
        self.inner.parent_beacon_block_root()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FocilPayloadBuilderAttributes {
    inner: EthPayloadBuilderAttributes,
    inclusion_list: Vec<Vec<u8>>,
}

impl PayloadBuilderAttributes for FocilPayloadBuilderAttributes {
    type RpcPayloadAttributes = FocilPayloadAttributes;
    type Error = Infallible;

    fn try_new(parent: B256, attributes: FocilPayloadAttributes) -> Result<Self, Infallible> {
        Ok(Self {
            inner: EthPayloadBuilderAttributes::new(parent, attributes.inner),
            inclusion_list: attributes.inclusion_list,
        })
    }

    fn payload_id(&self) -> PayloadId {
        self.inner.id
    }

    fn parent(&self) -> B256 {
        self.inner.parent
    }

    fn timestamp(&self) -> u64 {
        self.inner.timestamp
    }

    fn parent_beacon_block_root(&self) -> Option<B256> {
        self.inner.parent_beacon_block_root
    }

    fn suggested_fee_recipient(&self) -> Address {
        self.inner.suggested_fee_recipient
    }

    fn prev_randao(&self) -> B256 {
        self.inner.prev_randao
    }

    fn withdrawals(&self) -> &Withdrawals {
        &self.inner.withdrawals
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[non_exhaustive]
pub struct FocilEngineTypes;

impl PayloadTypes for FocilEngineTypes {
    type BuiltPayload = EthBuiltPayload;
    type PayloadAttributes = FocilPayloadAttributes;
    type PayloadBuilderAttributes = FocilPayloadBuilderAttributes;
}

impl EngineTypes for FocilEngineTypes {
    type ExecutionPayloadV1 = ExecutionPayloadV1;
    type ExecutionPayloadV2 = ExecutionPayloadEnvelopeV2;
    type ExecutionPayloadV3 = ExecutionPayloadEnvelopeV3;
    type ExecutionPayloadV4 = ExecutionPayloadEnvelopeV4;
}

#[derive(Debug, Clone)]
pub struct FocilEngineValidator {
    chain_spec: Arc<ChainSpec>,
}

impl<T> EngineValidator<T> for FocilEngineValidator
where
    T: EngineTypes<PayloadAttributes = FocilPayloadAttributes>,
{
    fn validate_version_specific_fields(
        &self,
        version: EngineApiMessageVersion,
        payload_or_attrs: PayloadOrAttributes<'_, T::PayloadAttributes>,
    ) -> Result<(), EngineObjectValidationError> {
        validate_version_specific_fields(&self.chain_spec, version, payload_or_attrs)
    }

    fn ensure_well_formed_attributes(
        &self,
        version: EngineApiMessageVersion,
        attributes: &T::PayloadAttributes,
    ) -> Result<(), EngineObjectValidationError> {
        validate_version_specific_fields(&self.chain_spec, version, attributes.into())?;

        // NOTE
        //
        // currently there is no engine validation specified for the inclusion list, since the CL
        // does not validate the inclusion lists that it observes upon receipt over gossip-sub.

        Ok(())
    }
}

#[derive(Debug, Default, Clone, Copy)]
#[non_exhaustive]
pub struct FocilEngineValidatorBuilder;

impl<N> EngineValidatorBuilder<N> for FocilEngineValidatorBuilder
where
    N: FullNodeTypes<Types: NodeTypesWithEngine<Engine = FocilEngineTypes, ChainSpec = ChainSpec>>,
{
    type Validator = FocilEngineValidator;

    async fn build_validator(self, ctx: &BuilderContext<N>) -> eyre::Result<Self::Validator> {
        Ok(FocilEngineValidator {
            chain_spec: ctx.chain_spec(),
        })
    }
}

#[derive(Debug, Default, Clone)]
#[non_exhaustive]
pub struct FocilPayloadServiceBuilder;

impl<Node, Pool> PayloadServiceBuilder<Node, Pool> for FocilPayloadServiceBuilder
where
    Node:
        FullNodeTypes<Types: NodeTypesWithEngine<Engine = FocilEngineTypes, ChainSpec = ChainSpec>>,
    Pool: TransactionPool + Unpin + 'static,
{
    async fn spawn_payload_service(
        self,
        ctx: &BuilderContext<Node>,
        pool: Pool,
    ) -> eyre::Result<PayloadBuilderHandle<<Node::Types as NodeTypesWithEngine>::Engine>> {
        let payload_builder = FocilPayloadBuilder::default();
        let conf = ctx.payload_builder_config();

        let payload_job_config = BasicPayloadJobGeneratorConfig::default()
            .interval(conf.interval())
            .deadline(conf.deadline())
            .max_payload_tasks(conf.max_payload_tasks())
            .extradata(conf.extradata_bytes());

        let payload_generator = BasicPayloadJobGenerator::with_builder(
            ctx.provider().clone(),
            pool,
            ctx.task_executor().clone(),
            payload_job_config,
            payload_builder,
        );
        let (payload_service, payload_builder) =
            PayloadBuilderService::new(payload_generator, ctx.provider().canonical_state_stream());

        ctx.task_executor()
            .spawn_critical("payload builder service", Box::pin(payload_service));

        Ok(payload_builder)
    }
}

fn cfg_and_block_env<EvmConfig>(
    evm_config: &EvmConfig,
    payload_config: &PayloadConfig<FocilPayloadBuilderAttributes>,
) -> (CfgEnvWithHandlerCfg, BlockEnv)
where
    EvmConfig: ConfigureEvm<Header = Header>,
{
    let next_attributes = NextBlockEnvAttributes {
        timestamp: payload_config.attributes.timestamp(),
        suggested_fee_recipient: payload_config.attributes.suggested_fee_recipient(),
        prev_randao: payload_config.attributes.prev_randao(),
    };
    evm_config.next_cfg_and_block_env(payload_config.parent_block.header.header(), next_attributes)
}

#[derive(Debug, Default, Clone)]
#[non_exhaustive]
pub struct FocilPayloadBuilder;

impl<Pool, Client> PayloadBuilder<Pool, Client> for FocilPayloadBuilder
where
    Client: StateProviderFactory + ChainSpecProvider<ChainSpec = ChainSpec>,
    Pool: TransactionPool,
{
    type Attributes = FocilPayloadBuilderAttributes;
    type BuiltPayload = EthBuiltPayload;

    fn try_build(
        &self,
        args: BuildArguments<Pool, Client, Self::Attributes, Self::BuiltPayload>,
    ) -> Result<BuildOutcome<Self::BuiltPayload>, PayloadBuilderError> {
        let BuildArguments {
            client,
            pool,
            mut cached_reads,
            config,
            cancel: _,
            best_payload,
        } = args;

        let chain_spec = client.chain_spec();
        let evm_config = EthEvmConfig::new(chain_spec.clone());
        let (cfg_env, block_env) = cfg_and_block_env(&evm_config, &config);

        let PayloadConfig {
            ref parent_block,
            extra_data,
            attributes,
        } = config;

        let state_provider = client.state_by_block_hash(config.parent_block.hash())?;
        let state = StateProviderDatabase::new(state_provider);
        let mut db = State::builder()
            .with_database_ref(cached_reads.as_db(state))
            .with_bundle_update()
            .build();

        debug!(target: "payload_builder", id=%attributes.payload_id(), parent_hash = ?parent_block.hash(), parent_number = parent_block.number, "building new payload");

        let block_gas_limit: u64 = block_env.gas_limit.to::<u64>();
        let base_fee = block_env.basefee.to::<u64>();
        let block_number = block_env.number.to::<u64>();

        let mut cumulative_gas_used = 0;
        let sum_blob_gas_used: u64 = 0;
        let mut total_fees = U256::ZERO;

        let mut executed_txs = Vec::new();
        let mut executed_senders = Vec::new();

        let mut system_caller = SystemCaller::new(&evm_config, chain_spec.clone());

        system_caller
            .pre_block_beacon_root_contract_call(
                &mut db,
                &cfg_env,
                &block_env,
                attributes.parent_beacon_block_root(),
            )
            .map_err(|err| {
                warn!(target: "payload_builder",
                    parent_hash=%parent_block.hash(),
                    %err,
                    "failed to apply beacon root contract call for payload"
                );
                PayloadBuilderError::Internal(err.into())
            })?;

        system_caller.pre_block_blockhashes_contract_call(
            &mut db,
            &cfg_env,
            &block_env,
            parent_block.hash(),
        )
        .map_err(|err| {
            warn!(target: "payload_builder", parent_hash=%parent_block.hash(), %err, "failed to update blockhashes for payload");
            PayloadBuilderError::Internal(err.into())
        })?;

        let mut receipts = Vec::new();

        // TODO
        //
        // add other block building logic (e.g. blobs) before/after inclusion list logic. currently
        // we only consider inclusion list transactions for the block.

        // NOTE
        //
        // it may make sense to put any inclusion list logic behind a prague fork check.

        let inclusion_list_txs: Vec<_> = attributes
            .inclusion_list
            .clone()
            .into_iter()
            .map(|tx| TransactionSignedEcRecovered::decode(&mut tx.as_slice()).ok())
            .collect();

        let mut i = 0;
        let n = inclusion_list_txs.len();

        // the inclusion list bitfield tracks whether we need to consider the inclusion list
        // transaction at the corresponding index any longer.
        //
        // if the tx was not properly encoded, then we mark it false.
        // if the tx gas limit would exceed the block gas limit, then we mark it false.
        // if the tx failed to execute for some reason that cannot change, then we mark it false.
        // if the tx executes successfully and is added to the block, then we mark it false.
        //
        // if a transaction from the inclusion list is executed successfully, then we need to go
        // back over each of the remaining inclusion list transactions that might now be valid.
        let mut inclusion_list_bitfield = vec![true; n];

        while i < n {
            if !inclusion_list_bitfield[i] {
                i += 1;
                continue;
            }

            // transaction not properly encoded
            let Some(ref tx) = inclusion_list_txs[i] else {
                inclusion_list_bitfield[i] = false;
                i += 1;
                continue;
            };

            // transaction gas limit too high
            if cumulative_gas_used + tx.gas_limit() > block_gas_limit {
                inclusion_list_bitfield[i] = false;
                i += 1;
                continue;
            }

            let env = EnvWithHandlerCfg::new_with_cfg_env(
                cfg_env.clone(),
                block_env.clone(),
                evm_config.tx_env(tx),
            );
            let mut evm = evm_config.evm_with_env(&mut db, env);

            let ResultAndState { result, state } = match evm.transact() {
                Ok(res) => res,
                Err(err) => match err {
                    EVMError::Transaction(err) => {
                        match err {
                            // a transaction whose nonce is too high may become valid.
                            // a transaction whose sender lacks funds may become valid.
                            InvalidTransaction::NonceTooHigh { .. }
                            | InvalidTransaction::LackOfFundForMaxFee { .. } => {}
                            _other => {
                                inclusion_list_bitfield[i] = false;
                            }
                        }

                        i += 1;
                        continue;
                    }
                    err => return Err(PayloadBuilderError::EvmExecutionError(err)),
                },
            };

            drop(evm);
            db.commit(state);

            let gas_used = result.gas_used();
            cumulative_gas_used += gas_used;

            #[allow(clippy::needless_update)]
            receipts.push(Some(Receipt {
                tx_type: tx.tx_type(),
                success: result.is_success(),
                cumulative_gas_used,
                logs: result.into_logs().into_iter().map(Into::into).collect(),
                ..Default::default()
            }));

            let miner_fee = tx
                .effective_tip_per_gas(Some(base_fee))
                .expect("fee is always valid; execution succeeded");
            total_fees += U256::from(miner_fee) * U256::from(gas_used);

            executed_senders.push(tx.signer());
            executed_txs.push(tx.clone().into_signed());

            inclusion_list_bitfield[i] = false;
            i = 0;
        }

        if !is_better_payload(best_payload.as_ref(), total_fees) {
            return Ok(BuildOutcome::Aborted {
                fees: total_fees,
                cached_reads,
            });
        }

        let (requests, requests_root) =
            if chain_spec.is_prague_active_at_timestamp(attributes.timestamp()) {
                let deposit_requests =
                    parse_deposits_from_receipts(&chain_spec, receipts.iter().flatten()).map_err(
                        |err| PayloadBuilderError::Internal(RethError::Execution(err.into())),
                    )?;
                let withdrawal_requests = system_caller
                    .post_block_withdrawal_requests_contract_call(&mut db, &cfg_env, &block_env)
                    .map_err(|err| PayloadBuilderError::Internal(err.into()))?;
                let consolidation_requests = system_caller
                    .post_block_consolidation_requests_contract_call(&mut db, &cfg_env, &block_env)
                    .map_err(|err| PayloadBuilderError::Internal(err.into()))?;

                let requests = [
                    deposit_requests,
                    withdrawal_requests,
                    consolidation_requests,
                ]
                .concat();
                let requests_root = calculate_requests_root(&requests);
                (Some(requests.into()), Some(requests_root))
            } else {
                (None, None)
            };

        let WithdrawalsOutcome {
            withdrawals_root,
            withdrawals,
        } = commit_withdrawals(
            &mut db,
            &chain_spec,
            attributes.timestamp(),
            attributes.withdrawals().clone(),
        )?;

        db.merge_transitions(BundleRetention::Reverts);

        let execution_outcome = ExecutionOutcome::new(
            db.take_bundle(),
            vec![receipts.clone()].into(),
            block_number,
            vec![requests.clone().unwrap_or_default()],
        );
        let receipts_root = execution_outcome
            .receipts_root_slow(block_number)
            .expect("Number is in range");
        let logs_bloom = execution_outcome
            .block_logs_bloom(block_number)
            .expect("Number is in range");

        let hashed_state = HashedPostState::from_bundle_state(&execution_outcome.state().state);
        let (state_root, trie_output) = {
            let state_provider = db.database.0.inner.borrow_mut();
            state_provider
                .db
                .state_root_with_updates(hashed_state.clone())
                .inspect_err(|err| {
                    warn!(target: "payload_builder",
                        parent_hash=%parent_block.hash(),
                        %err,
                        "failed to calculate state root for payload"
                    );
                })?
        };

        let transactions_root = proofs::calculate_transaction_root(&executed_txs);

        let mut blob_sidecars = Vec::new();
        let mut excess_blob_gas = None;
        let mut blob_gas_used = None;

        if chain_spec.is_cancun_active_at_timestamp(attributes.timestamp()) {
            blob_sidecars = pool.get_all_blobs_exact(
                executed_txs
                    .iter()
                    .filter(|tx| tx.is_eip4844())
                    .map(|tx| tx.hash)
                    .collect(),
            )?;

            excess_blob_gas = if chain_spec.is_cancun_active_at_timestamp(parent_block.timestamp) {
                let parent_excess_blob_gas = parent_block.excess_blob_gas.unwrap_or_default();
                let parent_blob_gas_used = parent_block.blob_gas_used.unwrap_or_default();
                Some(calc_excess_blob_gas(
                    parent_excess_blob_gas,
                    parent_blob_gas_used,
                ))
            } else {
                Some(calc_excess_blob_gas(0, 0))
            };

            blob_gas_used = Some(sum_blob_gas_used);
        }

        let header = Header {
            parent_hash: parent_block.hash(),
            ommers_hash: EMPTY_OMMER_ROOT_HASH,
            beneficiary: block_env.coinbase,
            state_root,
            transactions_root,
            receipts_root,
            withdrawals_root,
            logs_bloom,
            timestamp: attributes.timestamp(),
            mix_hash: attributes.prev_randao(),
            nonce: BEACON_NONCE.into(),
            base_fee_per_gas: Some(base_fee),
            number: parent_block.number + 1,
            gas_limit: block_gas_limit,
            difficulty: U256::ZERO,
            gas_used: cumulative_gas_used,
            extra_data,
            parent_beacon_block_root: attributes.parent_beacon_block_root(),
            blob_gas_used: blob_gas_used.map(Into::into),
            excess_blob_gas: excess_blob_gas.map(Into::into),
            requests_root,
        };

        let block = Block {
            header,
            body: BlockBody {
                transactions: executed_txs,
                ommers: vec![],
                withdrawals,
                requests,
            },
        };

        let sealed_block = block.seal_slow();
        debug!(target: "payload_builder", ?sealed_block, "sealed built block");

        let executed = ExecutedBlock {
            block: Arc::new(sealed_block.clone()),
            senders: Arc::new(executed_senders),
            execution_output: Arc::new(execution_outcome),
            hashed_state: Arc::new(hashed_state),
            trie: Arc::new(trie_output),
        };

        let mut payload = EthBuiltPayload::new(
            attributes.payload_id(),
            sealed_block,
            total_fees,
            Some(executed),
        );
        payload.extend_sidecars(blob_sidecars);

        Ok(BuildOutcome::Better {
            payload,
            cached_reads,
        })
    }

    fn build_empty_payload(
        &self,
        client: &Client,
        config: PayloadConfig<Self::Attributes>,
    ) -> Result<Self::BuiltPayload, PayloadBuilderError> {
        let PayloadConfig {
            parent_block,
            extra_data,
            attributes,
        } = config;
        let chain_spec = client.chain_spec();
        <EthereumPayloadBuilder as PayloadBuilder<Pool, Client>>::build_empty_payload(
            &EthereumPayloadBuilder::new(EthEvmConfig::new(chain_spec.clone())),
            client,
            PayloadConfig {
                parent_block,
                extra_data,
                attributes: attributes.inner,
            },
        )
    }
}

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
struct FocilNode;

impl NodeTypes for FocilNode {
    type Primitives = ();
    type ChainSpec = ChainSpec;
}

impl NodeTypesWithEngine for FocilNode {
    type Engine = FocilEngineTypes;
}

impl<N> Node<N> for FocilNode
where
    N: FullNodeTypes<Types: NodeTypesWithEngine<Engine = FocilEngineTypes, ChainSpec = ChainSpec>>,
{
    type ComponentsBuilder = ComponentsBuilder<
        N,
        EthereumPoolBuilder,
        FocilPayloadServiceBuilder,
        EthereumNetworkBuilder,
        EthereumExecutorBuilder,
        EthereumConsensusBuilder,
        FocilEngineValidatorBuilder,
    >;
    type AddOns = EthereumAddOns;

    fn components_builder(&self) -> Self::ComponentsBuilder {
        ComponentsBuilder::default()
            .node_types::<N>()
            .pool(EthereumPoolBuilder::default())
            .payload(FocilPayloadServiceBuilder::default())
            .network(EthereumNetworkBuilder::default())
            .executor(EthereumExecutorBuilder::default())
            .consensus(EthereumConsensusBuilder::default())
            .engine_validator(FocilEngineValidatorBuilder::default())
    }

    fn add_ons(&self) -> Self::AddOns {
        EthereumAddOns::default()
    }
}

#[derive(Debug, Clone, Copy, Default, clap::Args)]
struct RethCliFocilExt {
    #[arg(long)]
    pub enable_ext: bool,
}

#[cfg_attr(not(test), rpc(server, namespace = "engine"))]
#[cfg_attr(test, rpc(server, client, namespace = "engine"))]
pub trait FocilExtApi {
    #[method(name = "inclusionList")]
    fn inclusion_list(&self) -> RpcResult<Vec<Vec<u8>>>;
}

pub struct FocilExt<Pool> {
    pool: Pool,
}

impl<Pool> FocilExtApiServer for FocilExt<Pool>
where
    Pool: TransactionPool + Clone + 'static,
{
    fn inclusion_list(&self) -> RpcResult<Vec<Vec<u8>>> {
        const MAX: usize = 16;

        let mut txs = self.pool.pending_transactions();
        txs.sort_by_key(|tx| tx.timestamp);
        txs.truncate(MAX);
        let txs: Vec<Vec<u8>> = txs
            .into_iter()
            .map(|tx| {
                let len = tx.encoded_length();
                let mut buf = vec![0u8; len];
                let tx = tx.to_recovered_transaction();
                tx.encode(&mut buf);
                buf
            })
            .collect();

        Ok(txs)
    }
}
