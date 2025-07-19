import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Завантаження даних
file_path = "comparison_results_avg.csv"
df = pd.read_csv(file_path)
df.columns = [col.strip() for col in df.columns]

sns.set(style="whitegrid")
palette = sns.color_palette("tab10")

# Фільтри для графіків ефективності
pop_size_fixed = 500
generations_fixed = 200

# Графік 1 - Прискорення Master Slave від розміру популяції
plt.figure(figsize=(8,5))
df_g1 = df[(df["Threads"] == 12) & (df["Generations"] == generations_fixed)]
sns.lineplot(data=df_g1, x="Pop.Size", y="MS Speedup", marker='o', color=palette[0])
plt.title("Прискорення Master Slave від розміру популяції\n(Threads=12, Generations=200)")
plt.xlabel("Розмір популяції (Pop.Size)")
plt.ylabel("Прискорення (MS Speedup)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Графік 2 - Прискорення Island Model від розміру популяції
plt.figure(figsize=(8,5))
df_g2 = df[(df["Threads"] == 12) & (df["Generations"] == generations_fixed)]
sns.lineplot(data=df_g2, x="Pop.Size", y="Island Speedup", marker='o', color=palette[1])
plt.title("Прискорення Island Model від розміру популяції\n(Threads=12, Generations=200)")
plt.xlabel("Розмір популяції (Pop.Size)")
plt.ylabel("Прискорення (Island Speedup)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Графік 3 - Прискорення Master Slave від поколінь
plt.figure(figsize=(8,5))
df_g3 = df[(df["Threads"] == 8) & (df["Pop.Size"] == 200)]
sns.lineplot(data=df_g3, x="Generations", y="MS Speedup", marker='o', color=palette[2])
plt.title("Прискорення Master Slave від кількості поколінь\n(Threads=8, Pop.Size=200)")
plt.xlabel("Кількість поколінь (Generations)")
plt.ylabel("Прискорення (MS Speedup)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Графік 4 - Прискорення Island Model від поколінь
plt.figure(figsize=(8,5))
df_g4 = df[(df["Threads"] == 8) & (df["Pop.Size"] == 200)]
sns.lineplot(data=df_g4, x="Generations", y="Island Speedup", marker='o', color=palette[3])
plt.title("Прискорення Island Model від кількості поколінь\n(Threads=8, Pop.Size=200)")
plt.xlabel("Кількість поколінь (Generations)")
plt.ylabel("Прискорення (Island Speedup)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Графік 5 - Ефективність Master Slave від кількості процесів (Threads), фіксовані Pop.Size і Generations
plt.figure(figsize=(8,5))
df_g5 = df[(df["Pop.Size"] == pop_size_fixed) & (df["Generations"] == generations_fixed)]
sns.lineplot(data=df_g5, x="Threads", y="MS Eff.", marker='o', color=palette[4])
plt.title(f"Ефективність Master Slave від кількості процесів\n(Pop.Size={pop_size_fixed}, Generations={generations_fixed})")
plt.xlabel("Кількість потоків (Threads)")
plt.ylabel("Ефективність (MS Eff.)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Графік 6 - Ефективність Island Model від кількості процесів (Threads), фіксовані Pop.Size і Generations
plt.figure(figsize=(8,5))
df_g6 = df[(df["Pop.Size"] == pop_size_fixed) & (df["Generations"] == generations_fixed)]
sns.lineplot(data=df_g6, x="Threads", y="Island Eff.", marker='o', color=palette[5])
plt.title(f"Ефективність Island Model від кількості процесів\n(Pop.Size={pop_size_fixed}, Generations={generations_fixed})")
plt.xlabel("Кількість потоків (Threads)")
plt.ylabel("Ефективність (Island Eff.)")
plt.grid(True)
plt.tight_layout()
plt.show()
