from TB2J.exchange import ExchangeNCL
from TB2J.exchange_qspace import ExchangeCLQspace
from TB2J.exchangeCL2 import ExchangeCL2

# NOTE: the GPU exchange classes are imported lazily inside select_exchange so
# that the default CPU import path (e.g. wannier90_interface -> Manager) does
# not pull in TB2J.gpu / JAX. JAX must remain an optional dependency.


class Manager:
    def __init__(self, atoms, models, basis, colinear, **kwargs):
        # computing exchange
        print("Starting to calculate exchange.")
        use_gpu = kwargs.get("use_gpu", False)
        ExchangeClass = self.select_exchange(colinear, use_gpu=use_gpu)

        output_path = kwargs.get("output_path", "TB2J_results")
        exchange = ExchangeClass(tbmodels=models, atoms=atoms, basis=basis, **kwargs)

        if use_gpu:
            exchange.run(
                path=output_path,
                use_gpu=use_gpu,
                vectorize_energy=kwargs.get("vectorize_energy", False),
                e_batch_size=kwargs.get("e_batch_size", None),
            )
        else:
            exchange.calculate_all()
            exchange.write_output(path=output_path)
        print(f"All calculation finished. The results are in {output_path} directory.")

    def select_exchange(self, colinear, qspace=False, use_gpu=False):
        if colinear:
            if qspace:
                return ExchangeCLQspace
            elif use_gpu:
                from TB2J.gpu.exchangeCL_gpu import ExchangeCL2GPU

                return ExchangeCL2GPU
            else:
                return ExchangeCL2
        else:
            if use_gpu:
                from TB2J.gpu.exchange_ncl_gpu import ExchangeNCLGPU

                return ExchangeNCLGPU
            else:
                return ExchangeNCL
