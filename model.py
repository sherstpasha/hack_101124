import torch.nn as nn


# ======== Определение и обучение классификатора ======== #
# Определение модели классификатора
class EmbeddingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EmbeddingClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes)
        )

    def forward(self, embedding):
        logits = self.classifier(embedding)
        return logits