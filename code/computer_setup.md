# How the comptuer will be set up on the computer runnign script.py


## Structure:
- High level structure will be the same, sweep.py sits in same place relative to this project

## Data:
- Preprocessing already complete


MORPH/datasets/normalized_revin/
    1dbe_pdebench/
        test/
            be1d_test.h5
        train/
            be1d_train.h5
        val/
            be1d_val.h5
    2dSW_pdebench/
        test/
            sw2d_test.h5
        train/
            sw2d_train.h5
        val/
            sw2d_val.h5
    DR2d_data_pdebench/
        test/
            dr2d_test.h5
        train/
            dr2d_train.h5
        val/
            dr2d_val.h5

## Models

MORPH/models/FM/
    morph_s_fm.pth
    morph_ti_fm.pth
    morph_l_fm.pth

## Checkpoints

Sweep A:
- No checkpoints, data only
- Temporarily save last checkpoint incase of failure

Sweep B:
- Save best checkpoint per dataset
- Temporarily save last checkpoint incase of failure

## Output

Metrics
- See plan.md Record section, ensure script logs each metric appropriately
- Per sweep:
- One file for overall sweep metrics; One sweep config per row
- One file for individual epoch steps; Sweep is one row then each row per is epoch stats
- stdout goes to a saved file

When either sweep is started, it should make a folder 
{proj_root}/
    code/
    MORPH/
    out/
        sweep_{A||B}_{DATETIME}/
            models/
                *CHECKPOINTS AND SAVED MODELS GO HERE - LABEL CORRECTLTY*
            sweep_metrics.csv
            epoch_metrics.csv