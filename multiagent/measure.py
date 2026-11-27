"""
Μέτρηση του multi-agent σε σετ ερωτήσεων, με ρυθμίσεις από τα configs/ (Hydra), δίπλα
στο σημερινό bot. Κάθε απάντηση γράφεται στο reports/multiagent_answers.csv, όπως με το
evaluate.py, και η μέτρηση συνεχίζει από εκεί που έμεινε αν διακοπεί.

    venv/bin/python -m multiagent.measure experiment=tev1-en-clean 'measure.sets=[holdout6]'
    venv/bin/python -m multiagent.measure -m +sweep=validator_threshold     σάρωση (configs/sweep/)

Σάρωση: όλοι οι συνδυασμοί των τιμών του configs/sweep/<όνομα>.yaml τρέχουν αυτόματα,
ο ένας μετά τον άλλο, όπως τα sweeps του Feathers. Η στήλη config κάθε γραμμής είναι οι
αλλαγές της εκτέλεσης (π.χ. «experiment=tev1-en-clean validator.threshold=0.7»), ή το
measure.name αν δοθεί. Πίνακες μετά:

    venv/bin/python -m multiagent.evaluate metrics --configs "<όνομα>,<όνομα>"

Θέλει το Ollama ανοιχτό.
"""
from __future__ import annotations

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from multiagent import ROOT, llm


@hydra.main(config_path=str(ROOT / "configs"), config_name="multiagent", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from multiagent import evaluate          # φορτώνει και το BERT της βαθμολόγησης

    name = cfg.measure.name or HydraConfig.get().job.override_dirname or "default"
    pipe = instantiate(cfg.pipeline)
    llm.require([m for m in (pipe.welcome.split_model, pipe.welcome.reform_model, pipe.security.model,
                             pipe.validator.model, pipe.validator.final_model) if m is not None])
    pipe.classifier.load()
    print(f"\n«{name}»\n{pipe.describe()}\n")
    evaluate.run_pipeline_on(pipe, name, list(cfg.measure.sets), sheet=cfg.measure.out, limit=cfg.measure.limit)


if __name__ == "__main__":
    main()
