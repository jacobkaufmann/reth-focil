use alloy::rlp::Encodable;
use clap::Parser;
use jsonrpsee::{core::RpcResult, proc_macros::rpc};
use reth::{chainspec::EthereumChainSpecParser, cli::Cli};
use reth_node_ethereum::EthereumNode;
use reth_transaction_pool::TransactionPool;

fn main() {
    Cli::<EthereumChainSpecParser, RethCliFocilExt>::parse()
        .run(|builder, args| async move {
            let handle = builder
                .node(EthereumNode::default())
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
