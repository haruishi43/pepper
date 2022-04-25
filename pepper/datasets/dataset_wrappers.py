#!/usr/bin/env python3

from mmcv.utils import print_log
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.
    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    add `get_cat_ids` function.
    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)
        self.separate_eval = separate_eval

        self.NUM_PIDS = datasets[0].NUM_PIDS

        if not separate_eval:
            if len(set([type(ds) for ds in datasets])) != 1:
                raise NotImplementedError(
                    "To evaluate a concat dataset non-separately, "
                    "all the datasets should have same types"
                )

    def evaluate(self, results, *args, indices=None, logger=None, **kwargs):
        """Evaluate the results.
        Args:
            results (list[list | tuple]): Testing results of the dataset.
            indices (list, optional): The indices of samples corresponding to
                the results. It's unavailable on ConcatDataset.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict[str: float]: AP results of the total dataset or each separate
            dataset if `self.separate_eval=True`.
        """
        if indices is not None:
            raise NotImplementedError(
                "Use indices to evaluate speific samples in a ConcatDataset "
                "is not supported by now."
            )

        assert len(results) == len(self), (
            "Dataset and results have different sizes: "
            f"{len(self)} v.s. {len(results)}"
        )

        # Check whether all the datasets support evaluation
        for dataset in self.datasets:
            assert hasattr(
                dataset, "evaluate"
            ), f"{type(dataset)} haven't implemented the evaluate function."

        if self.separate_eval:
            total_eval_results = dict()
            for dataset_idx, dataset in enumerate(self.datasets):
                start_idx = (
                    0
                    if dataset_idx == 0
                    else self.cumulative_sizes[dataset_idx - 1]
                )
                end_idx = self.cumulative_sizes[dataset_idx]

                results_per_dataset = results[start_idx:end_idx]
                print_log(
                    f"Evaluateing dataset-{dataset_idx} with "
                    f"{len(results_per_dataset)} images now",
                    logger=logger,
                )

                eval_results_per_dataset = dataset.evaluate(
                    results_per_dataset, *args, logger=logger, **kwargs
                )
                for k, v in eval_results_per_dataset.items():
                    total_eval_results.update({f"{dataset_idx}_{k}": v})

            return total_eval_results
        else:
            original_data_infos = self.datasets[0].data_infos
            self.datasets[0].data_infos = sum(
                [dataset.data_infos for dataset in self.datasets], []
            )
            eval_results = self.datasets[0].evaluate(
                results, logger=logger, **kwargs
            )
            self.datasets[0].data_infos = original_data_infos
            return eval_results
