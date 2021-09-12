import matplotlib.pyplot as plt

EPOCHS = 20
total_loss_epoch_sgd = [2.3050, 1.9892, 1.7120, 1.5780, 1.4671, 1.4037, 1.3144, 1.3966, 1.2389, 1.2005, 1.0962, 1.2299, 1.0570, 1.0020, 0.9975, 0.9275, 0.9342, 0.8692, 0.9160, 0.8504]
# [2.3050, 2.3048, 2.3046, 2.3044, 2.3042, 2.3040, 2.3039, 2.3037, 2.3035, 2.3033, 2.3032, 2.3030, 2.3028, 2.3027, 2.3025, 2.3023, 2.3022, 2.3020, 2.3018, 2.3016]
total_loss_epoch_svrg = [2.3050, 1.9888, 1.7215, 1.5735, 1.4534, 1.3228, 1.2400, 1.1709, 1.1149, 1.1948, 1.1325, 1.0067, 1.0213, 0.9466, 0.8945, 0.8869, 0.8355, 0.8305, 0.8247, 0.8457]
# [2.3050, 2.3048, 2.3046, 2.3044, 2.3042, 2.3040, 2.3039, 2.3037, 2.3035, 2.3033, 2.3032, 2.3030, 2.3028, 2.3027, 2.3025, 2.3023, 2.3022, 2.3020, 2.3018, 2.3016]
total_loss_epoch_projected_gd = [2.3050, 1.6269, 1.4483, 1.2939, 1.2321, 1.1175, 1.0550, 0.9947, 0.9193, 0.9001, 0.9224, 0.8777, 0.7964, 0.7524, 0.7284, 0.7295, 0.6352, 0.6675, 0.6430, 0.7056]
total_loss_epoch_projected_gd_shuffle = [2.3050, 1.6246, 1.4309, 1.2905, 1.2256, 1.1901, 1.0564, 1.0272, 0.9512, 0.8855, 0.8799, 0.8518, 0.8230, 0.7868, 0.7837, 0.7024, 0.7340, 0.6809, 0.6009, 0.6117]
# total_loss_epoch_adam = [2.3050, 1.7503, 1.6260, 1.5522, 1.5265, 1.4686, 1.4223, 1.3824, 1.3613, 1.3599, 1.3022, 1.2861, 1.2718, 1.2486, 1.2460, 1.2228, 1.1940, 1.1831, 1.1775, 1.1573]
total_loss_epoch_sag = [2.3050, 2.0822, 1.7654, 1.5634, 1.5067, 1.3841, 1.3432, 1.2997, 1.2122, 1.2169, 1.1724, 1.0929, 1.1896, 1.1033, 1.0013, 1.1011, 1.0360, 0.9343, 0.8995, 0.9505]
total_loss_epoch_projected_sgd = [2.3050, 2.2837, 1.9295, 1.6424, 1.5018, 1.4025, 1.3251, 1.2691, 1.2241, 1.1589, 1.1284, 1.0983, 1.0596, 1.0308, 1.0039, 0.9733, 0.9657, 0.9418, 0.9133, 0.8947]


test_accuracy_epoch_sgd = [28.2700, 36.6900, 42.3800, 45.8700, 49.2100, 51.5400, 48.5700, 53.7800, 54.5500, 58.1000, 53.3200, 59.1100, 60.7300, 59.5700, 61.7100, 61.5400, 62.9700, 61.9700, 63.0400, 63.2100]
# [10.1100, 10.1200, 10.1500, 10.1600, 10.1800, 10.2000, 10.2600, 10.2800, 10.2500, 10.2400, 10.2500, 10.2700, 10.3100, 10.3400, 10.3700, 10.4400, 10.4700, 10.4800, 10.5100, 10.5100]
test_accuracy_epoch_svrg = [28., 36.6100, 42.4400, 47.9200, 51.9200, 54.6800, 56.4500, 57.8900, 56.2500, 57.5000, 60.1000, 59.3400, 61.1700, 62.3800, 61.9800, 63.1900, 62.4400, 62.3000, 61.1900, 63.]
# [10.1100, 10.1300, 10.1500, 10.1600, 10.1800, 10.2000, 10.2500, 10.2800, 10.2500, 10.2400, 10.2500, 10.2600, 10.3100, 10.3400, 10.3700, 10.4100, 10.4600, 10.4800, 10.5100, 10.5100]
test_accuracy_epoch_projected_gd = [41.0900, 47.9300, 52.4400, 53.9500, 56.7000, 58.7600, 58.6900, 60.5300, 60.3000, 58.8800, 59.4100, 60.7400, 61.5700, 61.0300, 60.4200, 62.1700, 60.9000, 61.1700, 58.9200, 60.7700]
test_accuracy_epoch_projected_gd_shuffle = [39.3000, 48.0100, 52.6000, 54.0300, 55.6700, 58.1900, 58.7600, 59.9700, 61.3500, 60.8100, 59.9700, 60.4700, 61.8600, 60.7200, 62.4000, 60.8200, 62.0500, 62.8100, 61.9300, 62.5500]
# test_accuracy_epoch_adam = [36.6600, 40.1400, 43.7800, 45.0800, 46.8600, 48.9000, 50.1600, 50.7100, 50.6700, 52.7300, 53.0400, 53.5200, 54.3700, 54.1000, 54.6800, 55.1900, 55.3100, 55.7600, 56.2700, 56.2800]
test_accuracy_epoch_sag = [23.8200, 35.8800, 42.8000, 45.1100, 50.2400, 51.0400, 52.2700, 54.8400, 54.6000, 55.9600, 58.3900, 54.6700, 55.7100, 60.1000, 56.3800, 57.8100, 60.3400, 61.3700, 59.3300, 60.0600]
test_accuracy_epoch_projected_sgd = [15.6200, 29.5500, 38.9400, 44.7100, 48.4800, 51.0500, 53.2800, 54.4000, 56.5000, 57.4500, 58.0500, 59.3400, 59.9100, 60.3600, 60.6700, 61.1000, 61.6600, 61.8800, 62.4600, 62.4600]


# plt.plot(range(EPOCHS), total_loss_epoch_sgd, lw=4, label="SGD")
# plt.plot(range(EPOCHS), total_loss_epoch_svrg, lw=4, label="SVRG")
plt.plot(range(EPOCHS), total_loss_epoch_projected_gd, lw=4, label="Projected GD")
plt.plot(range(EPOCHS), total_loss_epoch_projected_gd_shuffle, lw=4, label="Projected GD(shuffle)")
# plt.plot(range(EPOCHS), total_loss_epoch_adam, lw=4, label="Adam")
# plt.plot(range(EPOCHS), total_loss_epoch_sag, lw=4, label="SAG")
# plt.plot(range(EPOCHS), total_loss_epoch_projected_sgd, lw=4, label="Projected SGD")

plt.yscale('log')
plt.xlabel('iteration')
plt.ylabel('loss_function')
plt.title('objective functions evolution(lr=0.01)')
plt.legend()
plt.savefig('lr001_loss_shuffle.png')
plt.show()


# plt.plot(range(EPOCHS), test_accuracy_epoch_sgd, lw=4, label="SGD")
# plt.plot(range(EPOCHS), test_accuracy_epoch_svrg, lw=4, label="SVRG")
plt.plot(range(EPOCHS), test_accuracy_epoch_projected_gd, lw=4, label="Projected GD")
plt.plot(range(EPOCHS), test_accuracy_epoch_projected_gd_shuffle, lw=4, label="Projected GD(shuffle)")
# plt.plot(range(EPOCHS), test_accuracy_epoch_adam, lw=4, label="Adam")
# plt.plot(range(EPOCHS), test_accuracy_epoch_sag, lw=4, label="SAG")
# plt.plot(range(EPOCHS), test_accuracy_epoch_projected_sgd, lw=4, label="Projected SGD")

plt.xlabel('iteration')
plt.ylabel('test_accuracy %')
plt.title('test accuracy evolution(lr=0.01)')
plt.legend()
plt.savefig('lr001_accuracy_shuffle.png')
plt.show()
