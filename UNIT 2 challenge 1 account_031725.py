class Bank_Account :
    def __init__(self, acc_num, acc_name, acc_bal=0.0):
        self._acc_num = acc_num
        self._acc_name = acc_name
        self._acc_bal = acc_bal

    def deposit(self, amount):
        if amount > 0:
            self._acc_bal += amount
            return f"Deposited ${amount}. New balance: ${self._acc_bal}"
        else:
            return "Invalid deposit amount."

    def withdraw(self, amount):
        if 0 < amount <= self._acc_bal:
            self._acc_bal -= amount
            return f"Withdrew ${amount}. New balance: ${self._acc_bal}"
        else:
            return "Insufficient funds or invalid withdrawal amount."

    def display_balance(self):
        return f"Account Balance for {self._acc_name}: ${self._acc_bal}"

# Testing the BankAccount class
if __name__ == "__main__":
    # Create an instance of BankAccount
    account = Bank_Account("123456", "John Doe", 1000.0)

    # Test deposit and withdrawal functionality
    A=int(input("ender deposited ammount "))
    print(account.display_balance())
    print(account.deposit(A))
    B=int(input("enter the withdrawal ammount "))
    print(account.withdraw(B))