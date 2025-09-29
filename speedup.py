import matplotlib.pyplot as plt
t_seq = 0.4694702625274658  # Из sequential.py
t_par2 = 0.41879844665527344   # Из 2 процессов
t_par4 = 0.41257405281066895  # Из 4 процессов
t_par8 = 0.43857526779174805   # Из 8 процессов
processes = [1, 2, 4, 8]
times = [t_seq, t_par2, t_par4, t_par8]
speedups = [times[0] / t for t in times]
plt.plot(processes, speedups, marker='o')
plt.xlabel('Number of processes')
plt.ylabel('Speedup')
plt.title('Speedup vs Number of Processes')
plt.grid(True)
plt.savefig('speedup_plot.png')

#python3 speedup.py