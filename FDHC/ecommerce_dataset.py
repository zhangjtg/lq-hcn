import json
import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ProductContext:
    """Environment-only context. Contains BOTH reserves."""
    product_id: int
    product_name: str
    seller_item_description: str
    init_price: int
    buyer_reserve_price: int
    seller_reserve_price: int
    quantity: float
    currency: str = "CNY"



@dataclass(frozen=True)
class SellerContext:
    """Seller-private view: contains seller reserve, NOT buyer reserve."""
    product_id: int
    product_name: str
    seller_item_description: str
    init_price: int
    seller_reserve_price: int
    currency: str = "CNY"


@dataclass(frozen=True)
class BuyerContext:
    """Buyer-private view: contains buyer reserve, NOT seller reserve."""
    product_id: int
    product_name: str
    seller_item_description: str
    init_price: int
    buyer_reserve_price: int
    currency: str = "CNY"


@dataclass(frozen=True)
class PublicContext:
    """Public view: contains NO reserves (used for opponent simulation inside seller MCTS)."""
    product_id: int
    product_name: str
    seller_item_description: str
    init_price: int
    currency: str = "CNY"


def load_products_json(path: str) -> List[ProductContext]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    products: List[ProductContext] = []
    for item in raw:
        products.append(
            ProductContext(
                product_id=int(item["product_id"]),
                product_name=str(item["product_name"]),
                seller_item_description=str(item["seller_item_description"]),
                init_price=int(item["init_price"]),
                buyer_reserve_price=int(item["buyer_reserve_price"]),
                seller_reserve_price=int(item["seller_reserve_price"]),
                currency=str(item.get("currency", "CNY")),
                quantity=float(item["quantity"]),
            )
        )
    return products


def sample_product(products: List[ProductContext], rng: Optional[random.Random] = None) -> ProductContext:
    r = rng or random
    return r.choice(products)


def as_seller_context(ctx: ProductContext) -> SellerContext:
    return SellerContext(
        product_id=ctx.product_id,
        product_name=ctx.product_name,
        seller_item_description=ctx.seller_item_description,
        init_price=ctx.init_price,
        seller_reserve_price=ctx.seller_reserve_price,
        currency=ctx.currency,
    )


def as_buyer_context(ctx: ProductContext) -> BuyerContext:
    return BuyerContext(
        product_id=ctx.product_id,
        product_name=ctx.product_name,
        seller_item_description=ctx.seller_item_description,
        init_price=ctx.init_price,
        buyer_reserve_price=ctx.buyer_reserve_price,
        currency=ctx.currency,
    )


def as_public_context(ctx: ProductContext) -> PublicContext:
    return PublicContext(
        product_id=ctx.product_id,
        product_name=ctx.product_name,
        seller_item_description=ctx.seller_item_description,
        init_price=ctx.init_price,
        currency=ctx.currency,
    )
