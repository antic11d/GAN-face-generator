import matplotlib.image as mpimg
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(8, 8))
columns = 2
rows = 2
for i in range(1, 5):
    path = f'faces/face-{2*i}_1.png'
    img = mpimg.imread(path)
    fig.add_subplot(rows, columns, i)
    plt.title(f'Epoha: {2*i}', fontsize=24)
    plt.imshow(img)
plt.savefig('faces.png')
plt.show()
