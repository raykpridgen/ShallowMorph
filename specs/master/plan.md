# Train MORPH to surrogate shallow water simulation

## Preprocessing
- should be almost no data cleaning, data appears well ordered
- Each group 0xxx can be used as a sequence
- Split groups into train, val, test 

## Training
- Show model full sequences of water evolution
- Fine tune final layer of nodes for this specific set
- Other layers remain frozen
- Need to define how many frames forward the model is predicting
- Uses a sliding window to account for each new prediction
- Need to define loss / error method
- Loss: per frame, or total sum of error from each frame in sequence?
- see train_usage.md for using the training algorithm


## Evaluation
- Keep comprehensive data on metrics as training runs
- Print best outcome at the end
- save model for inference
- separate script / mode of same script for infering and manually evaluating the saved model

## Visualization

### Already have a basic script to make a gif of evolution
- ensure this can be used on a real sequence as well as a given / predicted sequence

### Single frame diff plot
- Plot one row of three frames
- Actual, predicted, diff

### entire sequence diff evolution 
- Plot 2 rows of progressing frames
- Top is actual, bottom is given / predicted
- May need to increase plot's step between timesteps to plot cleanly
