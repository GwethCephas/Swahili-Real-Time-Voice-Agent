"""Semantic Kernel native plugin(s) for the Swahili customer-support agent.

Convention: `@kernel_function` names and descriptions stay in ENGLISH, because
that is what GPT-4o reads when deciding which function to call -- English
descriptions route more reliably than Swahili ones would. The *returned*
payload strings are Swahili, since that is what the caller actually hears.
"""
from __future__ import annotations

from typing import Annotated

from semantic_kernel.functions import kernel_function

# Mock data standing in for a real order/account backend.
_MOCK_ORDERS: dict[str, dict[str, str]] = {
    "12345": {"status": "imesafirishwa", "eta": "2026-09-27"},
    "67890": {"status": "inatayarishwa", "eta": "2026-09-29"},
}

_MOCK_ACCOUNTS: dict[str, float] = {
    "ACC-001": 15250.00,
    "ACC-002": 320.50,
}

_OPENING_HOURS: dict[str, str] = {
    "Jumatatu-Ijumaa": "08:00 - 18:00",
    "Jumamosi": "09:00 - 14:00",
    "Jumapili": "Tumefungwa",
}


class SwahiliCustomerSupportPlugin:
    """Customer-support tools exposed to the kernel; results are in Swahili."""

    @kernel_function(
        name="check_order_status",
        description="Check the shipping status of a customer order by order ID.",
    )
    def check_order_status(
        self,
        order_id: Annotated[str, "The order ID to look up, e.g. '12345'."],
    ) -> Annotated[str, "Order status message in Swahili."]:
        order = _MOCK_ORDERS.get(order_id)
        if not order:
            return f"Samahani, sikuweza kupata agizo lenye namba {order_id}. Tafadhali hakiki namba ya agizo."
        return (
            f"Agizo namba {order_id} liko katika hatua ya '{order['status']}'. "
            f"Linatarajiwa kufika tarehe {order['eta']}."
        )

    @kernel_function(
        name="check_account_balance",
        description="Check a customer's account balance by account ID.",
    )
    def check_account_balance(
        self,
        account_id: Annotated[str, "The account ID to look up, e.g. 'ACC-001'."],
    ) -> Annotated[str, "Account balance message in Swahili."]:
        balance = _MOCK_ACCOUNTS.get(account_id)
        if balance is None:
            return f"Samahani, sikuweza kupata akaunti yenye namba {account_id}."
        return f"Salio la akaunti {account_id} ni Shilingi {balance:,.2f}."

    @kernel_function(
        name="get_opening_hours",
        description="Get the store's opening hours for all days of the week.",
    )
    def get_opening_hours(self) -> Annotated[str, "Opening hours message in Swahili."]:
        lines = [f"{day}: {hours}" for day, hours in _OPENING_HOURS.items()]
        return "Saa za ufunguzi wa duka: " + "; ".join(lines) + "."
