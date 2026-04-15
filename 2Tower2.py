import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, util
from sentence_transformers.evaluation import SentenceEvaluator

# 1. Custom Evaluator to produce the 6 MSE Columns
class MultiDimensionMseEvaluator(SentenceEvaluator):
    def __init__(self, test_df, anchors_dict):
        super().__init__()
        self.sentences = test_df['comment'].tolist()
        self.anchors_dict = anchors_dict
        self.test_df = test_df

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        model.eval()
        metrics = {}
        all_mses = []
        
        # Encode all test comments once for speed
        comment_embeddings = model.encode(self.sentences, convert_to_tensor=True, show_progress_bar=False)
        
        # Define the 6 headers exactly as in your screenshot
        dimensions = [
            ("Level Of Justification", "level_of_justification"),
            ("Respect Towards Demands", "respect_towards_demands"),
            ("Respect Towards Counterarguments", "respect_towards_counterarguments"),
            ("Content Of Justification", "content_of_justification"),
            ("Respect Towards Groups", "respect_towards_groups"),
            ("Constructive Politics", "constructive_politics")
        ]

        for name, column_key in dimensions:
            anchor_desc = self.anchors_dict[column_key]
            anchor_embedding = model.encode(anchor_desc, convert_to_tensor=True, show_progress_bar=False)
            
            # AI Similarity Prediction
            predictions = util.cos_sim(comment_embeddings, anchor_embedding).flatten()
            
            # Real Labels (Normalized 0 to 1)
            targets = torch.tensor([(float(s) + 1) / 2 for s in self.test_df[column_key]]).to(model.device)
            
            # Calculate MSE for this specific column
            mse = torch.mean((predictions - targets) ** 2).item()
            metrics[f"Mse {name}"] = mse
            all_mses.append(mse)
            
        metrics["Mse Overall"] = np.mean(all_mses)
        return metrics

# 2. Data Loading & Cleaning
print("Loading dataset...")
dataset = load_dataset("lanretto/discourse_quality", data_files="full_labeled_flattened.csv", split="train")
dataset = dataset.filter(lambda x: x['comment'] is not None and len(str(x['comment'])) > 10)
split = dataset.train_test_split(test_size=1000, seed=42)

anchors = {
    "level_of_justification": "Logical depth and structural integrity of reasoning.",
    "respect_towards_demands": "Interpersonal tone toward the opinion or proposal in the parent comment.",
    "respect_towards_counterarguments": "Fairness and reciprocity when engaging with opposing views, such as steelmanning versus strawmanning.",
    "content_of_justification": "Scope of benefit invoked, ranging from narrow or tribal interests to the universal common good.",
    "respect_towards_groups": "Tone toward social identity groups mentioned in the comment.",
    "constructive_politics": "Orientation toward conflict resolution versus escalation."
}

# 3. Prepare Multi-Task Training/Validation Data
def prepare_data(batch):
    s1, s2, labels = [], [], []
    for i in range(len(batch["comment"])):
        for col in anchors.keys():
            s1.append(batch["comment"][i])
            s2.append(anchors[col])
            labels.append((float(batch[col][i]) + 1) / 2)
    return {"sentence1": s1, "sentence2": s2, "label": labels}

train_dataset = split["train"].map(prepare_data, batched=True, remove_columns=split["train"].column_names)
eval_dataset = split["test"].map(prepare_data, batched=True, remove_columns=split["test"].column_names)

# 4. Initialize Model and Custom Evaluator
model = SentenceTransformer('all-mpnet-base-v2')
custom_evaluator = MultiDimensionMseEvaluator(pd.DataFrame(split["test"]), anchors)

# 5. Training Arguments
args = SentenceTransformerTrainingArguments(
    output_dir="discourse_mse_production",
    num_train_epochs=2,
    per_device_train_batch_size=64,   # Lowered to fit in shared memory
    gradient_accumulation_steps=2,    # Maintains the 'effective' batch size of 128
    fp16=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,                
    save_strategy="steps",
    save_steps=200,               
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="Mse Overall",
    greater_is_better=False
)



# 6. Trainer Setup
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,     # This generates the "Validation Loss" column
    loss=losses.CosineSimilarityLoss(model),
    evaluator=custom_evaluator      # This generates the 7 MSE columns
)

# 7. Start Training
trainer.train()
