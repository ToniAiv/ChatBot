"""
Συζήτηση με το multi-agent σύστημα στο τερματικό, με ρυθμίσεις από τα configs/ (Hydra).

    venv/bin/python -m multiagent.chat                                    συζήτηση
    venv/bin/python -m multiagent.chat 'once="θέλω βεβαίωση κατοικίας"'   ένα μήνυμα και τέλος
    venv/bin/python -m multiagent.chat experiment=tev1-en-clean           μια ρύθμιση που έχουμε μετρήσει
    venv/bin/python -m multiagent.chat security.scope=before              αλλαγή μίας τιμής
    venv/bin/python -m multiagent.chat model@security.model=qwen3_4b      άλλο μοντέλο για τον Security
    venv/bin/python -m multiagent.chat print_config=true                  τύπωσε όλο το config στην αρχή

Κάθε agent και κάθε μοντέλο φτιάχνεται χωριστά από το δικό του αρχείο
(configs/multiagent.yaml). Κάθε εκτέλεση κρατά το πλήρες config της στο
outputs/chat/<ημερομηνία_ώρα>/.hydra/.

Μέσα στη συζήτηση: /νεα ξεχνά το αίτημα που περιμένει διευκρίνιση, /τελος κλείνει.
Θέλει το Ollama ανοιχτό.
"""
from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from multiagent import ROOT
from multiagent.cli import converse


def shown(cfg: DictConfig) -> str:
    """Το config σε YAML, με τις τιμές λυμένες. Το pipeline δείχνει μόνο ποιους agents
    παίρνει (${welcome} κ.λπ.), για να μην τυπωθούν δύο φορές."""
    out = OmegaConf.to_container(cfg, resolve=True)
    out["pipeline"] = OmegaConf.to_container(cfg.pipeline, resolve=False)
    return OmegaConf.to_yaml(OmegaConf.create(out))


@hydra.main(config_path=str(ROOT / "configs"), config_name="multiagent", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.print_config:
        print(shown(cfg))
    converse(instantiate(cfg.pipeline), cfg.once)


if __name__ == "__main__":
    main()
