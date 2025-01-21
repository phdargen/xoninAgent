import { CdpAction, CdpActionSchemaAny } from "./cdp_action";
import { DeployNftAction } from "./deploy_nft";
import { DeployTokenAction } from "./deploy_token";
import { GetBalanceAction } from "./get_balance";
import { GetBalanceNftAction } from "./get_balance_nft";
import { GetWalletDetailsAction } from "./get_wallet_details";
import { MintNftAction } from "./mint_nft";
import { RegisterBasenameAction } from "./register_basename";
import { RequestFaucetFundsAction } from "./request_faucet_funds";
import { TradeAction } from "./trade";
import { TransferAction } from "./transfer";
import { TransferNftAction } from "./transfer_nft";
import { WrapEthAction } from "./wrap_eth";
import { WOW_ACTIONS } from "./defi/wow";
import { PYTH_ACTIONS } from "./data/pyth";

/**
 * Retrieves all CDP action instances.
 * WARNING: All new CdpAction classes must be instantiated here to be discovered.
 *
 * @returns - Array of CDP action instances
 */
export function getAllCdpActions(): CdpAction<CdpActionSchemaAny>[] {
  return [
    new GetWalletDetailsAction(),
    new DeployNftAction(),
    new DeployTokenAction(),
    new GetBalanceAction(),
    new GetBalanceNftAction(),
    new MintNftAction(),
    new RegisterBasenameAction(),
    new RequestFaucetFundsAction(),
    new TradeAction(),
    new TransferAction(),
    new TransferNftAction(),
    new WrapEthAction(),
  ];
}

export const CDP_ACTIONS = getAllCdpActions().concat(WOW_ACTIONS).concat(PYTH_ACTIONS);

export {
  CdpAction,
  CdpActionSchemaAny,
  GetWalletDetailsAction,
  DeployNftAction,
  DeployTokenAction,
  GetBalanceAction,
  GetBalanceNftAction,
  MintNftAction,
  RegisterBasenameAction,
  RequestFaucetFundsAction,
  TradeAction,
  TransferAction,
  TransferNftAction,
  WrapEthAction,
};
