import dlib #face detection, landmark detection, face allignment
import matplotlib.pyplot as plt #이미지 띄어줄것
import matplotlib.patches as patches
import tensorflow as tf # model 사용
import numpy as np


##############LOAD MODELS######################
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('BeautyGAN-master/BeautyGAN-master/models/shape_predictor_5_face_landmarks.dat') #랜드마크 5개짜리 모델 사용.


###############LOAD IMAGES#########################3
img = dlib.load_rgb_image('BeautyGAN-master/BeautyGAN-master/imgs/01.jpg')

plt.figure(figsize=(16, 10))
plt.imshow(img)


##################FIND FACES#######################
img_result = img.copy()

dets = detector(img, 1) #얼굴 찾아서 얼굴 위치한 rectangle 반환
#print(dets)

if len(dets) == 0:
    print('cannot find faces!')

fig, ax = plt.subplots(1, figsize=(16, 10))
#fig란 figure로써 - 전체 subplot을 말한다. ex) 서브플로안에 몇개의 그래프가 있던지 상관없이  그걸 담는 하나.
#전체 사이즈를 말한다.
#print(fig)

#ax는 axe로써 - 전체 중 낱낱개를 말한다 ex) 서브플롯 안에 2개(a1,a2)의 그래프가 있다면 a1, a2 를 일컬음
#print(ax)
for det in dets:
    x, y, w, h = det.left(), det.top(), det.width(), det.height()

    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none') #matplot patches의 사각형그리기
    ax.add_patch(rect)

ax.imshow(img_result)



#######################FIND LANDMARKS 5POINTS########################3
fig, ax = plt.subplots(1, figsize=(16, 10))

objs = dlib.full_object_detections() #dlib의 full_object_detections 클래스 initialization => 추후 얼굴 수평으로 맞출 때 사용할것.

for detection in dets:
    s = sp(img, detection) #sp() : 얼굴의 랜드마크를 찾는다. img를 넣어주고 rectangle(얼굴)의 위치를 넣어주면 shape가 나옴. => s
    objs.append(s)
    #print(s)

    for point in s.parts():
        #print(point)
        circle = patches.Circle((point.x, point.y), radius=3, edgecolor='r', facecolor='r')
        ax.add_patch(circle)

ax.imshow(img_result)


#######################ALIGN FACES#######################
faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)

fig, axes = plt.subplots(1, len(faces)+1, figsize=(20, 16)) #subplots(nrows=1, ncols=len(faces)+1)

axes[0].imshow(img)

for i, face in enumerate(faces):
    axes[i+1].imshow(face)


########################FUNTIONALIZE#####################
def align_faces(img):
    dets = detector(img, 1)

    objs = dlib.full_object_detections()

    for detection in dets:
        s = sp(img, detection)
        objs.append(s)

    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)

    return faces

# test
test_img = dlib.load_rgb_image('BeautyGAN-master/BeautyGAN-master/imgs/02.jpg')

test_faces = align_faces(test_img)

fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(20, 16))
axes[0].imshow(test_img)

for i, face in enumerate(test_faces):
    axes[i+1].imshow(face)


#########################LOAD BeautyGAN Pretained#################
sess = tf.compat.v1.Session()

sess.run(tf.compat.v1.global_variables_initializer())

saver = tf.compat.v1.train.import_meta_graph('BeautyGAN-master/BeautyGAN-master/models/model.meta') #모델의 그래프를 불러온다.
saver.restore(sess, tf.train.latest_checkpoint('BeautyGAN-master/BeautyGAN-master/models')) #모델의 weights를 로드한다.
graph = tf.compat.v1.get_default_graph()

X = graph.get_tensor_by_name('X:0') # source #그래프에서 노드의 이름으로 텐서를 불러온다.
Y = graph.get_tensor_by_name('Y:0') # reference
Xs = graph.get_tensor_by_name('generator/xs:0') # output



########################Preprocess and Postprocess Functions###############
def preprocess(img): #이미지 전처리 0-255 값을 => -1 ~ 1 값으로 바꿈.
    return img.astype(np.float32) / 127.5 - 1.

def postprocess(img): #전처리한 값을 0-255 값으로 돌림.
    return ((img + 1.) * 127.5).astype(np.uint8)


#######################Load Images#######################
img1 = dlib.load_rgb_image('BeautyGAN-master/BeautyGAN-master/imgs/12.jpg')
img1_faces = align_faces(img1)

img2 = dlib.load_rgb_image('BeautyGAN-master/BeautyGAN-master/imgs/makeup/XMY-014.png')
img2_faces = align_faces(img2)

fig, axes = plt.subplots(1, 2, figsize=(16, 10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])

######################Run#######################
src_img = img1_faces[0]
ref_img = img2_faces[0]

X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0)

Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)

output = sess.run(Xs, feed_dict={
    X: X_img,
    Y: Y_img
})

output_img = postprocess(output[0])

fig, axes = plt.subplots(1, 3, figsize=(20, 10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)
plt.show() #이걸 해줘야 figure 창이 뜸. ipynb와 다름.
