import os
import json
import asyncio
from datetime import datetime
    
# CURRENCY_TOKEN = "💰"
CURRENCY_TOKEN = "$"
CURRENCY_TOKEN_SHORT = "$"
STARTING_BALANCE = 100

class CurrencyManager:
    """Manages player currency balances across the Discord server"""
    
    def __init__(self, data_dir="data"):
        """Initialize the currency manager with a data directory"""

        self.data_dir = data_dir
        self.currency_file = os.path.join(data_dir, "currency.json")
        self.inventory_file = os.path.join(data_dir, "inventory.json")
        self.transactions_file = os.path.join(data_dir, "transactions.json")
        self.balances = {}
        self.inventories = {}
        self.lock = asyncio.Lock()
        self.token = CURRENCY_TOKEN
        self.token_short = CURRENCY_TOKEN_SHORT
        self.starting_balance = STARTING_BALANCE
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        self._load_data()


    def amount_string(self, amount, short=False):
        """Format an amount as a string with the currency token"""
        
        return f"{self.token if not short else self.token_short}{amount}"
    

    def _load_data(self):
        """Load currency data from file"""

        if os.path.exists(self.currency_file):
            try:
                with open(self.currency_file, 'r') as f:
                    self.balances = json.load(f)
                self.balances = {int(k): v for k, v in self.balances.items()}
            except json.JSONDecodeError:
                print(f"Error loading currency data. Starting with empty balances.")
                self.balances = {}
        else:
            self.balances = {}

        if os.path.exists(self.inventory_file):
            try:
                with open(self.inventory_file, 'r') as f:
                    self.inventories = json.load(f)
                self.inventories = {int(k): v for k, v in self.inventories.items()}
            except json.JSONDecodeError:
                print(f"Error loading inventory data. Starting with empty inventories.")
                self.inventories = {}
        else:
            self.inventories = {}
    

    async def _save_data(self):
        """Save currency data to file"""

        async with self.lock:
            with open(self.currency_file, 'w') as f:
                json.dump(self.balances, f, indent=2)

            with open(self.inventory_file, 'w') as f:
                json.dump(self.inventories, f, indent=2)
    

    async def _log_transaction(self, user_id, amount, transaction_type, game=None, details=None):
        """Log a transaction to the transactions file"""

        async with self.lock:
            transactions = []
            if os.path.exists(self.transactions_file):
                try:
                    with open(self.transactions_file, 'r') as f:
                        transactions = json.load(f)
                except json.JSONDecodeError:
                    transactions = []
            
            transaction = {
                "user_id": user_id,
                "amount": amount,
                "type": transaction_type,
                "timestamp": datetime.now().isoformat(),
                "game": game,
                "details": details
            }
            
            transactions.append(transaction)
            
            with open(self.transactions_file, 'w') as f:
                json.dump(transactions, f, indent=2)
    

    async def get_balance(self, user_id):
        """Get a player's current balance"""

        user_id = int(user_id)

        # AI user
        if user_id < 0:
            return 9999999999

        if user_id not in self.balances:
            self.balances[user_id] = self.starting_balance
        return self.balances.get(user_id, 0)
    

    async def add_funds(self, user_id, amount, game=None, details=None):
        """Add funds to a player's account"""
        
        if amount <= 0:
            return False, "Amount must be positive"
        
        user_id = int(user_id)
        
        async with self.lock:
            if user_id not in self.balances:
                self.balances[user_id] = 0
            
            self.balances[user_id] += amount
        
        await self._save_data()
        await self._log_transaction(user_id, amount, "credit", game, details)
        
        return True, self.balances[user_id]
    

    async def remove_funds(self, user_id, amount, game=None, details=None):
        """Remove funds from a player's account"""
        if amount <= 0:
            return False, "Amount must be positive"
        
        user_id = int(user_id)
        current_balance = await self.get_balance(user_id)
        
        if current_balance < amount:
            return False, f"Insufficient funds (balance: {CURRENCY_TOKEN}{current_balance}, needed: {CURRENCY_TOKEN}{amount})"
        
        async with self.lock:
            self.balances[user_id] -= amount
        
        await self._save_data()
        await self._log_transaction(user_id, amount, "debit", game, details)
        
        return True, self.balances[user_id]
    

    async def transfer_funds(self, from_user_id, to_user_id, amount, game=None, details=None):
        """Transfer funds between two player accounts"""
        success, result = await self.remove_funds(from_user_id, amount, game, f"Transfer to {to_user_id}")
        
        if not success:
            return False, result
        
        await self.add_funds(to_user_id, amount, game, f"Transfer from {from_user_id}")
        
        return True, f"Transferred {CURRENCY_TOKEN}{amount} from {from_user_id} to {to_user_id}"
    

    async def get_inventory(self, user_id):
        """Get a player's inventory"""
        
        user_id = int(user_id)
        return self.inventories.get(user_id, {})
    

    async def add_item(self, user_id, item, quantity=1, game=None, details=None, item_type=None):
        user_id = int(user_id)

        async with self.lock:
            if user_id not in self.inventories:
                self.inventories[user_id] = {}

            if item not in self.inventories[user_id]:
                self.inventories[user_id][item] = {}
            
            self.inventories[user_id][item]['count'] = self.inventories[user_id][item].get('count', 0) + quantity
            self.inventories[user_id][item]['game'] = game
            self.inventories[user_id][item]['details'] = details
            self.inventories[user_id][item]['type'] = item_type

        await self._save_data()
        return True, f"Added {quantity} {item} to {user_id}'s inventory"
    

    async def ensure_minimum_balance(self, user_id, minimum=100):
        """Ensure a player has at least the minimum balance (for new players)"""

        user_id = int(user_id)
        current_balance = await self.get_balance(user_id)
        
        if current_balance < minimum:
            amount_to_add = minimum - current_balance
            
            await self.add_funds(user_id, amount_to_add, "system", "Starting balance")
            
            return True, f"Added {CURRENCY_TOKEN}{amount_to_add} starting balance"
        
        return False, "Balance already meets minimum"
    

    async def get_leaderboard(self, limit=10):
        """Get the top players by balance"""

        sorted_balances = sorted(self.balances.items(), key=lambda x: x[1], reverse=True)
        return sorted_balances[:limit]
    

    async def get_trophies(self, limit=10, item_type='trophy'):
        """Get the top players by inventory trophy count"""

        sorted_inventories = sorted(self.inventories.items(), key=lambda x: sum([v['count'] for v in x[1].values() if v['type'] == item_type]), reverse=True)
        return sorted_inventories[:limit]
    

    async def reset_balance(self, user_id, new_balance=0):
        """Reset a player's balance (admin function)"""

        user_id = int(user_id)
        
        async with self.lock:
            old_balance = self.balances.get(user_id, 0)
            self.balances[user_id] = new_balance
        
        await self._save_data()
        await self._log_transaction(
            user_id, 
            new_balance - old_balance, 
            "reset", 
            "admin", 
            f"Balance reset from {CURRENCY_TOKEN}{old_balance} to {CURRENCY_TOKEN}{new_balance}"
        )
        
        return True, f"Reset balance for user {user_id} to {CURRENCY_TOKEN}{new_balance}"
