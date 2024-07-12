from TB2J.exchange import ExchangeNCL
from TB2J.exchange_qspace import ExchangeCLQspace
from TB2J.exchangeCL2 import ExchangeCL2


class Manager:
    def __init__(self, atoms, models, basis, colinear, **kwargs):
        # computing exchange
        print("Starting to calculate exchange.")
        ExchangeClass = self.select_exchange(colinear)
        exchange = ExchangeClass(tbmodels=models, atoms=atoms, basis=basis, **kwargs)
        output_path = kwargs.get("output_path", "TB2J_results")
        exchange.run(path=output_path)
        print(f"All calculation finished. The results are in {output_path} directory.")

    def select_exchange(self, colinear, qspace=False):
        if colinear:
            if qspace:
                return ExchangeCLQspace
            else:
                return ExchangeCL2
        else:
            return ExchangeNCL
