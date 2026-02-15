"""Quick privacy checks for the 'cannot be accessed' constraints."""

from ecommerce_dataset import ProductContext, as_seller_context, as_buyer_context, as_public_context
from Game import NegotiationGame
from llm_client import LLMClient


def main():
    ctx = ProductContext(
        product_id=1,
        product_name="Smartphone",
        seller_item_description="Test",
        init_price=2999,
        buyer_reserve_price=2700,
        seller_reserve_price=2600,
        currency="CNY",
    )

    sctx = as_seller_context(ctx)
    bctx = as_buyer_context(ctx)
    pctx = as_public_context(ctx)

    assert not hasattr(sctx, "buyer_reserve_price"), "SellerContext should NOT have buyer_reserve_price"
    assert not hasattr(bctx, "seller_reserve_price"), "BuyerContext should NOT have seller_reserve_price"
    assert not hasattr(pctx, "buyer_reserve_price") and not hasattr(pctx, "seller_reserve_price"), "PublicContext should have no reserves"

    game = NegotiationGame(
        seller_llm=LLMClient(mode="heuristic"),
        buyer_llm=LLMClient(mode="heuristic"),
    )

    st = game.get_initial_state(pctx)

    assert 2700 not in st.tolist(), "Buyer reserve leaked into state!"
    assert 2600 not in st.tolist(), "Seller reserve leaked into state!"

    # Seller moves with SellerContext (no buyer reserve available)
    _ = game.neural_valid_moves(st, sctx)

    # Transition to seller turn (buyer bids first)
    st2 = game.get_next_state(st, (0, 2500), player=-1)
    st2 = game.change_perspective(st2, 1)

    # Seller candidate generation should not crash and should not access buyer reserve
    _ = game.neural_valid_moves(st2, sctx)

    # Buyer offer should never exceed buyer reserve in BuyerContext mode
    st3 = game.get_next_state(st2, (0, 3000), player=1)
    st3 = game.change_perspective(st3, -1)
    act = game.buyer_offer(st3, bctx)
    assert act[1] <= ctx.buyer_reserve_price, "Buyer offer exceeded buyer reserve!"

    # Public-sim buyer offers should not exceed displayed init_price
    cands = game.buyer_candidate_offers(st3, pctx, k=5)
    for _, price in cands:
        assert price <= ctx.init_price, "Public-sim buyer offer exceeded displayed price!"

    print("Privacy checks passed.")


if __name__ == "__main__":
    main()
