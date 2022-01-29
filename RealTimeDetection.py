import cv2
import numpy as np
import torch



cam = cv2.VideoCapture(0)

if not cam.isOpened():
  print ("Could not open cam")
  exit()

while(1):
    ret, frame = cam.read()
    if ret:
        
        frame = cv2.flip(frame,1)        
        ROI_frame = frame[100:100+480*3, 200:200+640*3].copy()
        ROI_small = cv2.resize(ROI_frame, (640,480))
        ROI_array = np.array(ROI_small, dtype=np.float32)
        ROI_tensor = torch.tensor(ROI_array)
        ROI_tensor = ROI_tensor.permute(2,0,1)
        ROI_tensor = ROI_tensor/torch.max(ROI_tensor)

        # display = cv2.rectangle(frame.copy(),(200,100),(500,400),(0,255,0),2)
        # label = sess.run('Y_PRED_CLS:0', feed_dict={'X_INPUT:0':ROI/255})
        # word = alphabets[int(label)]
        # cv2.putText(display, word, (200, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100,180,12), 2)
        # cv2.imshow('curFrame',display)
        cv2.imshow('Roi', ROI_small)
         
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()