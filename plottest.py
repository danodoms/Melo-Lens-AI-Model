import matplotlib.pyplot as plt
import numpy as np

# Data for both cases: MobileNetV2 and MobileNetV2Unbalanced
classes = ['Healthy', 'Nitrogen', 'Phosphorus', 'Potassium']
f1_mobile_net_v2 = [0.26, 0.20, 0.27, 0.26]
f1_mobile_net_v2_unbalanced = [0.54, 0.17, 0.18, 0.09]

precision_mobile_net_v2 = [0.27, 0.21, 0.28, 0.24]
precision_mobile_net_v2_unbalanced = [0.55, 0.18, 0.17, 0.09]

recall_mobile_net_v2 = [0.25, 0.19, 0.27, 0.29]
recall_mobile_net_v2_unbalanced = [0.53, 0.17, 0.18, 0.10]

# Creating subplots for F1 score, precision, and recall comparison
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# F1 Score Plot
axs[0].bar(classes, f1_mobile_net_v2, width=0.4, label='MobileNetV2', align='center', color='skyblue')
axs[0].bar(classes, f1_mobile_net_v2_unbalanced, width=0.4, label='MobileNetV2 Unbalanced', align='edge', color='salmon')
axs[0].set_title('F1 Score Comparison')
axs[0].set_ylabel('F1 Score')
axs[0].legend()

# Precision Plot
axs[1].bar(classes, precision_mobile_net_v2, width=0.4, label='MobileNetV2', align='center', color='skyblue')
axs[1].bar(classes, precision_mobile_net_v2_unbalanced, width=0.4, label='MobileNetV2 Unbalanced', align='edge', color='salmon')
axs[1].set_title('Precision Comparison')
axs[1].set_ylabel('Precision')

# Recall Plot
axs[2].bar(classes, recall_mobile_net_v2, width=0.4, label='MobileNetV2', align='center', color='skyblue')
axs[2].bar(classes, recall_mobile_net_v2_unbalanced, width=0.4, label='MobileNetV2 Unbalanced', align='edge', color='salmon')
axs[2].set_title('Recall Comparison')
axs[2].set_ylabel('Recall')

# Display the plot
plt.tight_layout()
plt.show()
