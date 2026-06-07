import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# CREAZIONE DATASET
# =========================

# valori x
X = torch.linspace(0, 500, 501).reshape(-1, 1)

# funzione target: y = x^2 + 1
Y = X**2 + 1

# =========================
# DEFINIZIONE RETE NEURALE
# =========================

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 168),
            nn.ReLU(),
            nn.Linear(168, 1)
        )

    def forward(self, x):
        return self.model(x)

# creazione modello
model = NeuralNet()

# =========================
# LOSS E OTTIMIZZATORE
# =========================

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.2
)

# =========================
# TRAINING
# =========================

num_epochs = 10000

for epoch in range(num_epochs):

    # forward
    predictions = model(X)

    # calcolo errore
    loss = criterion(predictions, Y)

    # reset gradienti
    optimizer.zero_grad()

    # backward
    loss.backward()

    # aggiornamento pesi
    optimizer.step()

    # stampa ogni 500 epoche
    if epoch % 500 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.6f}")

# =========================
# TEST MODELLO
# =========================

with torch.no_grad():
    predicted = model(X)

# stampa di alcuni valori predetti
print(model(torch.tensor([[0.],[1.],[2.],[3.],[4.],[5.],[6.],[7.],[8.],[9.],[100.],[500.]])))

# =========================
# GRAFICO
# =========================

sns.set( style="whitegrid", font_scale=1.5 )

fig, ax = plt.subplots(figsize = (13, 7))

ax.set_facecolor("1")
ax.grid(True, color="Black", linewidth=0.5)

ax.plot(X.numpy(), Y.numpy(), linestyle="-", linewidth=4, color="#1b9e77", label="Funzione esatta")
ax.plot(X.numpy(), predicted.numpy(), linestyle="-", linewidth=4, color="#d95f02", label="Funzione predetta")

ax.yaxis.set_major_locator(plt.MultipleLocator(50000))

ax.set_title("Confronto tra funzione e esatta e funzione predetta dalla rete", pad=20, fontsize=20, fontweight="bold")

ax.legend()
plt.show()
