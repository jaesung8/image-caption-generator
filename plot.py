import pickle as pkl

with open('train_data.pkl', 'rb') as f:
    train_data = pkl.load(f)

# train_data = {
#                 "train_loss_list": train_loss_list,
#                 "train_top1_acc": train_top1_acc,
#                 "train_top5_acc": train_top5_acc,
#                 "valid_loss_list": valid_loss_list,
                # "valid_top1_acc": valid_top1_acc,
                # "valid_top5_acc": valid_top5_acc,
#             }


import matplotlib.pyplot as plt


plt.title('Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.plot(train_data['train_loss_list'])
plt.savefig('./train_loss.png')
plt.cla()

plt.title('Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.plot(train_data['valid_loss_list'])
plt.savefig('./valid_loss.png')
plt.cla()

plt.title('Trainging Top-5')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
plt.plot(train_data['train_top5_acc'])
plt.savefig('./train_top5.png')
plt.cla()

plt.title('Validation Top-5')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
plt.plot(train_data['valid_top5_acc'])
plt.savefig('./valid_top5.png')
plt.cla()

plt.title('Trainging Top-1')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
plt.plot(train_data['train_top1_acc'])
plt.savefig('./train_topp1.png')
plt.cla()

plt.title('Validation Top-1')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
plt.plot(train_data['valid_top1_acc'])
plt.savefig('./valid_top1.png')
plt.cla()