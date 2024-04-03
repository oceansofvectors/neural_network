import pandas as pd
import numpy as np
from nn import load

# Step 1: Load the test dataset
df_test = pd.read_csv('data/test.csv')


X_test = df_test.values / 255.0  # Normalize the data

model = load(filename='model_epoch_49.ckpt')  # Load the serialized model


test_predictions = model.forward(X_test)
predicted_labels = np.argmax(test_predictions, axis=1)

# Step 4: Create the submission DataFrame
submission = pd.DataFrame({
    "ImageId": range(1, len(predicted_labels) + 1),
    "Label": predicted_labels
})

# Step 5: Save the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)

print("Submission file has been created successfully.")
