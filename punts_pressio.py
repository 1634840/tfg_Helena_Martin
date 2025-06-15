import cv2
import os
from utils import vis
from model import HRpose, RESpose, ChainedPredictions, PyraNet, StackedHourGlass, PoseAttention
from skimage import io 
import numpy as np
import torchvision.transforms as transforms
import torch 
import utils.utils_ds as ut_ds
import utils.utils as ut
from core.inference import get_max_preds
import matplotlib.pyplot as plt
import math
import matplotlib.lines 
import json
from config import dict_sizes, dict_means, dict_stds, flip_pairs, joints_name, kps_lines
from skimage.filters.rank import median
from skimage.morphology import disk



def load_model(imgtype):
    """ Loading the  trained model"""
    n_jt = 14
    in_ch=1
    if imgtype=='RGB':
        in_ch=3
    
    model = HRpose.get_pose_net(in_ch=in_ch, out_ch=n_jt)
    
    
    #pthfile = (r'C:\Users\Asus\Desktop\TFG\barreja modelAI + codi SLP\output\SLP_IR_u12_HRpose_ts1\model_dump\final_state.pth')
    pthfile = (r'C:\Users\Asus\Desktop\TFG\SLP_IR_u12_HRpose_exp\model_dump\checkpoint.pth')
    #pthfile=(r'C:\Users\Asus\Desktop\TFG\SLP_IR_u12_RESpose_exp\SLP_IR_u12_RESpose_exp\model_dump\checkpoint.pth')
    #pthfile=(r'C:\Users\Asus\Desktop\TFG\SLP_IR_u12_ChainedPredictions_exp\SLP_IR_u12_ChainedPredictions_exp\model_dump\checkpoint.pth')
    #pthfile = (r'C:\Users\Asus\Desktop\TFG\SLP_IR_u12_PyraNet_exp\SLP_IR_u12_PyraNet_exp\model_dump\checkpoint.pth')    
    # = (r'C:\Users\Asus\Desktop\TFG\SLP_IR_u12_StackedHourGlass_exp\SLP_IR_u12_StackedHourGlass_exp\model_dump\checkpoint.pth')
    #pthfile = (r'C:\Users\Asus\Desktop\TFG\SLP_IR_u12_PoseAttention_exp\SLP_IR_u12_PoseAttention_exp\model_dump\checkpoint.pth')
    #model.load_state_dict(torch.load(pthfile, map_location=torch.device('cpu')))
    
    """
    model.load_state_dict(torch.load(pthfile, map_location=torch.device('cpu')))
    model.eval()#set to evaluation mode
    return model
    """ 
    checkpoint = torch.load(pthfile, map_location=torch.device('cpu'))
    
    if 'best_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['best_state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    
        
    model.eval()#set to evaluation mode
    return model
    
def enhance_image(img, apply_denoise=False):
    # Make sure it's grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)

    # Optional: Apply median filter to reduce noise
    if apply_denoise:
        img_clahe = median(img_clahe, disk(1))

    return img_clahe

    


def loadimgformat(image_path, imgtype): #input image size 576x1024
    """ Preparing the input image """
    img = io.imread(image_path)
    img = np.array(img) #numpy data 
    print("0. input image shape: ", img.shape)
    if imgtype=='IR': #IR image after reading has 2 dimension
        img = img[..., None] #add one dimension #Not for 'RGB'  

    scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False
    bbox=calculatebbox(dict_sizes[imgtype])
    print("1. input image shape: ", img.shape)
    img_patch, trans = ut_ds.generate_patch_image(img, bbox=bbox, do_flip=do_flip, scale=scale, rot=rot, do_occlusion=do_occlusion, input_shape=(256,256))   # ori to bb, flip first trans later

    if img_patch.ndim<3:
        img_channels = 1        # add one channel
        img_patch = img_patch[..., None]
    else:
        img_channels = img_patch.shape[2]   # the channels
    for i in range(img_channels):
        img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)

    mean=dict_means[imgtype]
    std=dict_stds[imgtype]    

    trans_tch = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)])
    pch_tch = trans_tch(img_patch)
    pch_tch = pch_tch.unsqueeze(0) # add batch dimension in this cas batch=1
    return pch_tch

def calculatebbox(size):
    """ Calculate the bounding box of image """
    #bounding box [a, b, c, d]
    b=0.0
    if size[1]>size[0]:
        a=-(size[1]-size[0])/2.0
        c=d=float(size[1])
    elif size[1]<size[0]:
        a=-(size[0]-size[1])/2.0
        c=d=float(size[0])
    else: 
        a=0.0
        c=d=float(size[0])
    bbox=[a,b,c,d]
    return bbox

def predict(model, img_tensor): #this img_tensor should be 4 dimension adapted to model 
    """ Predicting of keypoints of skeleton """
    with torch.no_grad():
        #print(img_tensor)
        input_flipped = img_tensor.flip(3).clone() 
        output_flipped = model(input_flipped)
        
        output_flipped = ut_ds.flip_back(output_flipped.cpu().numpy(), flip_pairs)
        #output_flipped = ut_ds.flip_back(output_flipped[0].cpu().numpy(), flip_pairs)
        output_flipped = torch.from_numpy(output_flipped.copy())
        #output_flipped = torch.from_numpy(output_flipped.copy()).unsqueeze(0) # N x n_jt xh x w tch #.copy()).cuda()
        
        output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]
        
        #print("output flipped: ", output_flipped)
        output = model(img_tensor)
        #if isinstance(output,list):
            #output = output[0]
        #print("output: ", output)
        output = (output + output_flipped) * 0.5
        #print("output final: ", output)
    return output, output_flipped

def predAndPrintImg(path,imgtype):
    """ This function reading the model and do the inference. 
    Later print the skeleton with the image original. """

    # model=load_model('RGB')
    model=load_model(imgtype)
    # inputimage='\SLP_Project\SLP-Code\image_000001.png'
    inputimage=path
    # inputimage='\SLP_Project\SLP-Code\image_000005.png'
    # inputimage='\SLP_Project\SLP_dataset\SLP\danaLab\\00020\RGB\\uncover\image_000042.png'
    # pred, input_tensor = predict(model,inputimage)
    img = loadimgformat(inputimage,imgtype)#img type tensor 4 dimension (1,3,256,256)
    # img = loadimgformatSimlab(inputimage)#img type tensor 4 dimension (1,3,256,256)
    pred, _ = predict(model, img) #pred type tensor 4 dimension (1,14,64,64)
    
    landmark = pred.cpu().numpy() # landmark type numpy 4 dimension (1,14,64,64)
    #pred conte les cordenades (x,y) de cada articulacio
    pred, _ = get_max_preds(landmark)#pred type numpy 3 dimension (1,14,2)
    
    # print(pred[0,:,0]*256/64-56)
    # print(pred[0,:,1]*256/64)
    pred2d_patch = np.ones((14, 3))
    pred2d_patch[:,:2] = pred / 64 * 256 
    # print(pred2d_patch) #print points of joints 
    # print("ATTENTION: ", pred2d_patch)
    mean=dict_means[imgtype]
    std=dict_stds[imgtype]

    #preparing the the input image for vis_keypoints function
    img_patch_vis = ut.ts2cv2(img[0], mean, std)#img_patch_vis type numpy 3 dimension (256,256,3)
   
    #add color map for IR image
    if imgtype == 'IR':
        img_patch_vis_color = cv2.applyColorMap(img_patch_vis, cv2.COLORMAP_HOT)
        # img_patch_vis = cv2.applyColorMap(img_patch_vis, cv2.dct_clrMap[imgtype])    
        tmpimgcolor = vis.vis_keypoints(img_patch_vis_color, pred2d_patch, kps_lines)
        cv2.namedWindow("output IR color", cv2.WINDOW_NORMAL) 
        cv2.imshow("output IR color", tmpimgcolor)
    
    tmpimg = vis.vis_keypoints(img_patch_vis, pred2d_patch, kps_lines)
    # Create window with freedom of dimensions
    cv2.namedWindow("output "+imgtype, cv2.WINDOW_NORMAL) 
    # Using cv2.imshow() method
    # Displaying the image
    cv2.imshow("output "+imgtype, tmpimg)

    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)
    # closing all open windows
    cv2.destroyAllWindows()
    #cv2.imwrite(f'{savedimgname}.png', tmpimg) #save the image 

    return img_patch_vis, pred2d_patch, tmpimg

def save_predictions(predictions, filename="predictions.json"):
    joints_name = [
        "R_Ankle", "R_Knee", "R_Hip", "L_Hip", "L_Knee", "L_Ankle", "R_Wrist", "R_Elbow", "R_Shoulder", "L_Shoulder",
        "L_Elbow", "L_Wrist", "Thorax", "Head"]
    
    data = {joints_name[i]: predictions[i].tolist() for i in range(len(joints_name))}
    
    with open(filename, "w") as f:
        json.dump(data, f)
    
    print(f"Prediccions guardades a {filename}")

def genDict(predictjoints):
    """  Creating diccionary of joints obtained after the inference"""
    dict_joints={}
    #initialate the diccionary with list 'joints_name' only keys
    for i in range(14): 
        dict_joints[joints_name[i]]=predictjoints[i]
    return dict_joints

#prediccio postura
def points_axis_center(dict_joints):
    """ Create the center points of shouder and hip. 
    (using for creating a center axis of human body) """
    d_s= (dict_joints['L_Shoulder'] + dict_joints['R_Shoulder']) / 2
    d_h= (dict_joints['L_Hip'] + dict_joints['R_Hip']) / 2
    return d_s, d_h
    
def create_line(p1,p2):
    """ Create the line with two pionts (through)"""
    x1=p1[0] 
    x2=p2[0]
    y1=p1[1]
    y2=p2[1] 
    
    if (x2-x1)!=0:
        slope = (y2 - y1) / (x2 - x1) #slope
        intercept = y1 - slope * x1 #intercept  
        xt=np.linspace(0, 255, 2)
        yt=slope*xt + intercept
    else:
        slope=0
        intercept = 0 #intercept 
        xt=np.array([x1, x1])
        yt = np.linspace(0, 255, 2)

    # plt.plot(xt, yt, 'r-')  # 'r-' for a red line
    return xt, yt, slope, intercept 
def distance_point_line(p, slope, intercept, threshold=0.0):
    """ Calculate the distance between a point and a line """
    x=p[0] 
    y=p[1]
    # Apply the formula to calculate the distance abs(y-ax-b=0)/sqrt((-a)**2+(1)**2)
    A = -slope 
    B = 1
    C = -intercept     
    distance = (A*x + B*y + C) / math.sqrt(A**2 + B**2) # distance = (A * x + B * y + C) / ((A ** 2 + B ** 2) ** 0.5)

    if distance > 0: 
        if slope > 0:  point_position = 0 #Left side of the linea 
        else: point_position = 1    
    elif distance < 0: 
        if slope > 0: point_position = 1 #Right side of the linea   
        else:  point_position = 0 
    else: 
        point_position = 2 #On the linea

    distance = abs(distance) 
    return distance, point_position

def distance_point_line_noSlope(p, xref, threshold=0.0):
    """ Calculate the distance between a point and a line when slope is not defined. 
    It means the line is parallel to the y-axis """
    x=p[0]   
    distance = x - xref
    if distance < 0: 
        point_position = 0 #left side of the linea
    elif distance > 0: 
        point_position = 1 #Right side of the linea
    else: 
        point_position = 2 #On the linea
     
    distance = abs(distance)
    return distance, point_position

def position_prediction(dict_joints): ###more than just prediction position of part inferior of body 
    """ Create a center axis of bady and predict the position """
    cnt_r=0
    cnt_l=0
    p1, p2 = points_axis_center(dict_joints)
    xt, yt, slope, intercept = create_line(p1,p2) ##when slope, intercept ==0, also need to implement another function
        
    if (p2[0]-p1[0])!=0:
        distance, point_position_R_Elbow = distance_point_line(dict_joints['R_Elbow'], slope, intercept, threshold=0.0)
        distance, point_position_L_Elbow = distance_point_line(dict_joints['L_Elbow'], slope, intercept, threshold=0.0)
        distance, point_position_R_Knee = distance_point_line(dict_joints['R_Knee'], slope, intercept, threshold=0.0)
        distance, point_position_L_Knee = distance_point_line(dict_joints['L_Knee'], slope, intercept, threshold=0.0)
    else:
        distance, point_position_R_Elbow = distance_point_line_noSlope(dict_joints['R_Elbow'], p1[0], threshold=0.0)
        distance, point_position_L_Elbow = distance_point_line_noSlope(dict_joints['L_Elbow'], p1[0], threshold=0.0)
        distance, point_position_R_Knee = distance_point_line_noSlope(dict_joints['R_Knee'], p1[0], threshold=0.0)
        distance, point_position_L_Knee = distance_point_line_noSlope(dict_joints['L_Knee'], p1[0], threshold=0.0)
    #----------------R_Elbow--------------       
    if (point_position_R_Elbow==0): cnt_l +=1
    elif(point_position_R_Elbow==1): cnt_r +=1
    else: 
        cnt_l=cnt_l
        cnt_r=cnt_r
    #----------------L_Elbow-------------- 
    if (point_position_L_Elbow==0): cnt_l +=1
    elif(point_position_L_Elbow==1): cnt_r +=1
    else: 
        cnt_l=cnt_l
        cnt_r=cnt_r
    #----------------R_Knee-------------- 
    if (point_position_R_Knee==0): cnt_l +=1
    elif(point_position_R_Knee==1): cnt_r +=1
    else: 
        cnt_l=cnt_l
        cnt_r=cnt_r
    #----------------L_Knee-------------- 
    if (point_position_L_Knee==0): cnt_l +=1
    elif(point_position_L_Knee==1): cnt_r +=1
    else: 
        cnt_l=cnt_l
        cnt_r=cnt_r


    cnt =  cnt_l - cnt_r
    print("contador position: ",cnt)
    position=0
    if cnt>0: 
        position=0
        print('left side on picture (right side on person)')
    elif cnt<0: 
        position=1
        print('right side on picture (left side on person)')
    else: 
        position=2
        print("Probably supina on picture")
    return xt, yt, position 

# prediccio pin points

def zona_cabezal_IR(dict_joints, position):
    #------------------OCCIPITAL, RIGHT EAR and LEFT EAR -------------------------
    pm=(dict_joints["Thorax"] + dict_joints["Head"]) /2
    pm[2]=0
    dict_joints["Occipital"]=pm.copy() 
    dict_joints["L_Ear"]=pm.copy()
    dict_joints["R_Ear"]=pm.copy()

    if position==0: #on picture is left side lying, on person is right side lying (pressure on right side)
        dict_joints['Head'][2]=0
        dict_joints['Thorax'][2]=0
        dict_joints["R_Ear"][2]=1
    elif position==1: #creating point r_ear
        dict_joints['Head'][2]=0
        dict_joints['Thorax'][2]=0
        dict_joints["L_Ear"][2]=1
    else:#creating point occipital
        dict_joints['Head'][2]=0
        dict_joints['Thorax'][2]=0   
        dict_joints["Occipital"][2]=1 
    # print(dict_joints["Occipital"],  dict_joints["R_Ear"], dict_joints["L_Ear"])
    return dict_joints

def zona_superior_cuerpo_IR(dict_joints, position):
    #------------------SHOULDER and HIP -------------------------
    if position == 2:
        dict_joints['L_Hip'][2]=1
        dict_joints['R_Hip'][2]=1
        dict_joints['L_Shoulder'][2]=1
        dict_joints['R_Shoulder'][2]=1
    elif position == 0:
        dict_joints['L_Hip'][2]=0
        dict_joints['R_Hip'][2]=1
        dict_joints['L_Shoulder'][2]=0
        dict_joints['R_Shoulder'][2]=1
    elif position == 1:
        dict_joints['L_Hip'][2]=1
        dict_joints['R_Hip'][2]=0
        dict_joints['L_Shoulder'][2]=1
        dict_joints['R_Shoulder'][2]=0
    #-------------------------------------------------------------------------------------
    
    
    #-----------------------------ELBOW and WRIST ------------in this cas not including the position Prone lying----------------------------
    #---Calculate necessaries--- 
    if (dict_joints['L_Shoulder'][0]-dict_joints['L_Hip'][0])!=0:
        xt, yt, slope, intercept = create_line(dict_joints['L_Shoulder'],dict_joints['L_Hip']) ##when slope, intercept ==0, also need to implement another function
        distance_l_elbow, point_position_L_Elbow = distance_point_line(dict_joints['L_Elbow'], slope, intercept, threshold=0.0)
        distance_l_wrist, point_position_L_Wrist = distance_point_line(dict_joints['L_Wrist'], slope, intercept, threshold=0.0) 
        dis_r_wrist_toLeftLine, poipos_r_wrist_toLeftLine = distance_point_line(dict_joints['R_Wrist'], slope, intercept, threshold=0.0)
    else:
        distance_l_elbow, point_position_L_Elbow = distance_point_line_noSlope(dict_joints['L_Elbow'], dict_joints['L_Shoulder'][0], threshold=0.0)
        distance_l_wrist, point_position_L_Wrist = distance_point_line_noSlope(dict_joints['L_Wrist'], dict_joints['L_Shoulder'][0], threshold=0.0)
        dis_r_wrist_toLeftLine, poipos_r_wrist_toLeftLine = distance_point_line_noSlope(dict_joints['R_Wrist'], dict_joints['L_Shoulder'][0], threshold=0.0)
    # -------------------------

    if (dict_joints['R_Shoulder'][0]-dict_joints['R_Hip'][0])!=0:
        xt, yt, slope1, intercept1 = create_line(dict_joints['R_Shoulder'],dict_joints['R_Hip']) ##when slope, intercept ==0, also need to implement another function
        distance_r_elbow, point_position_R_Elbow = distance_point_line(dict_joints['R_Elbow'], slope1, intercept1, threshold=0.0)
        distance_r_wrist, point_position_R_Wrist = distance_point_line(dict_joints['R_Wrist'], slope1, intercept1, threshold=0.0)
        dis_l_wrist_toRightLine, poipos_l_wrist_toRightLine = distance_point_line(dict_joints['L_Wrist'], slope1, intercept1, threshold=0.0)
    else:
        distance_r_elbow, point_position_R_Elbow = distance_point_line_noSlope(dict_joints['R_Elbow'], dict_joints['R_Shoulder'][0], threshold=0.0)
        distance_r_wrist, point_position_R_Wrist = distance_point_line_noSlope(dict_joints['R_Wrist'], dict_joints['R_Shoulder'][0], threshold=0.0)
        dis_l_wrist_toRightLine, poipos_l_wrist_toRightLine = distance_point_line_noSlope(dict_joints['L_Wrist'], dict_joints['R_Shoulder'][0], threshold=0.0)

    # print("Relation about L_Elbow with the line: ", point_position_L_Elbow)
    # print("Relation about L_Wrist with the line: ", point_position_L_Wrist)
    # print("Relation about R_Elbow with the line: ", point_position_R_Elbow)
    # print("Relation about R_Wrist with the line: ", point_position_R_Wrist)

    if (dict_joints['Head'][0]-dict_joints['L_Shoulder'][0])!=0:
        xt, yt, slope2, intercept2 = create_line(dict_joints['Head'],dict_joints['L_Shoulder']) ##when slope, intercept ==0, also need to implement another function
        # plt.plot(xt,yt)
        dist_l_w, position_L_Wrist = distance_point_line(dict_joints['L_Wrist'], slope2, intercept2, threshold=0.0)
        dist_l_e, position_L_Elbow = distance_point_line(dict_joints['L_Elbow'], slope2, intercept2, threshold=0.0)
    else:
        dist_l_w, position_L_Wrist = distance_point_line_noSlope(dict_joints['L_Wrist'], dict_joints['L_Shoulder'][0], threshold=0.0)
        dist_l_e, position_L_Elbow = distance_point_line_noSlope(dict_joints['L_Elbow'], dict_joints['L_Shoulder'][0], threshold=0.0)
    
    if (dict_joints['Head'][0]-dict_joints['R_Shoulder'][0])!=0:
        xt, yt, slope3, intercept3 = create_line(dict_joints['Head'],dict_joints['R_Shoulder']) 
        dist_r_w, position_R_Wrist = distance_point_line(dict_joints['R_Wrist'], slope3, intercept3, threshold=0.0)
        dist_r_e, position_R_Elbow = distance_point_line(dict_joints['R_Elbow'], slope3, intercept3, threshold=0.0)
    else:
        dist_r_w, position_R_Wrist = distance_point_line_noSlope(dict_joints['R_Wrist'], dict_joints['R_Shoulder'][0], threshold=0.0)
        dist_r_e, position_R_Elbow = distance_point_line_noSlope(dict_joints['R_Elbow'], dict_joints['R_Shoulder'][0], threshold=0.0)

    #---Cheking R_Elbow and L_Elbow---
    #-----------R_Elbow----------------------
    if (dict_joints["R_Elbow"][1]>dict_joints["R_Shoulder"][1]):
        if (point_position_R_Elbow==0) and (distance_r_elbow>7): 
            dict_joints["R_Elbow"][2]=1
    else:
        if (point_position_R_Elbow==0):
            dict_joints["R_Elbow"][2]=1
        else:
            if(position_R_Elbow==0) and (is_point_in_circle(dict_joints['R_Elbow'][:2], dict_joints["Head"][:2], 7)==False) and (is_point_in_circle(dict_joints['R_Elbow'][:2], dict_joints["R_Shoulder"][:2], 7)==False):
                dict_joints["R_Elbow"][2]=1
   
    #-----------L_Elbow----------------------
    if (dict_joints["L_Elbow"][1]>dict_joints["L_Shoulder"][1]):
        if (point_position_L_Elbow==1) and (distance_l_elbow>7): 
            dict_joints["L_Elbow"][2]=1
    else:
        if (point_position_L_Elbow==1):
            dict_joints["L_Elbow"][2]=1
        else:
            if(position_L_Elbow==1) and (is_point_in_circle(dict_joints['L_Elbow'][:2], dict_joints["Head"][:2], 7)==False) and (is_point_in_circle(dict_joints['L_Elbow'][:2], dict_joints["L_Shoulder"][:2], 7)==False):
                dict_joints["L_Elbow"][2]=1


    # #-----------R_Wrist----------------------
    # dis_r_wrist_toLeftLine, poipos_r_wrist_toLeftLine = distance_point_line(dict_joints['R_Wrist'], slope, intercept, threshold=0.0)
    if (dict_joints["R_Wrist"][1] > dict_joints["R_Shoulder"][1]) and (dict_joints["R_Wrist"][1] < dict_joints["R_Hip"][1]):
        if (point_position_R_Wrist==0) and (distance_r_wrist>7): 
            dict_joints["R_Wrist"][2]=1
        else:
            if(poipos_r_wrist_toLeftLine==1) and (dis_r_wrist_toLeftLine>7):
                dict_joints["R_Wrist"][2]=1

    elif (dict_joints["R_Wrist"][1]<=dict_joints["R_Shoulder"][1]):
        if (point_position_R_Wrist==0):
            dict_joints["R_Wrist"][2]=1
        else:
            if(position_R_Wrist==0) and (is_point_in_circle(dict_joints['R_Wrist'][:2], dict_joints["Head"][:2], 7)==False) and (is_point_in_circle(dict_joints['R_Wrist'][:2], dict_joints["R_Shoulder"][:2], 7)==False) and (is_point_in_circle(dict_joints['R_Wrist'][:2], dict_joints["R_Elbow"][:2], 7)==False):
                dict_joints["R_Wrist"][2]=1
    else:
        if (dict_joints['R_Hip'][0]-dict_joints['R_Knee'][0])!=0:
            _, _, slo, inter = create_line(dict_joints['R_Hip'],dict_joints['R_Knee']) 
            d_r_w, p_R_Wrist = distance_point_line(dict_joints['R_Wrist'], slo, inter, threshold=0.0)
        else:
            d_r_w, p_R_Wrist = distance_point_line_noSlope(dict_joints['R_Wrist'], dict_joints['R_Hip'][0], threshold=0.0)
        
        if (p_R_Wrist==0) and (d_r_w>7):
            dict_joints["R_Wrist"][2]=1
    
    # # #-----------L_Wrist----------------------
    # # dis_l_wrist_toRightLine, poipos_l_wrist_toRightLine = distance_point_line(dict_joints['L_Wrist'], slope1, intercept1, threshold=0.0)
    if (dict_joints["L_Wrist"][1]>dict_joints["L_Shoulder"][1]) and (dict_joints["L_Wrist"][1]<dict_joints["L_Hip"][1]):
        if (point_position_L_Wrist==0) and (distance_l_wrist>7): 
            dict_joints["L_Wrist"][2]=1
        else:
            if(poipos_l_wrist_toRightLine==1) and (dis_l_wrist_toRightLine>7):
                dict_joints["L_Wrist"][2]=1
        
    elif (dict_joints["L_Wrist"][1]<=dict_joints["L_Shoulder"][1]):
        if (point_position_L_Wrist==0):
            dict_joints["L_Wrist"][2]=1
        else:
            if(position_L_Wrist==0) and (is_point_in_circle(dict_joints['L_Wrist'][:2], dict_joints["Head"][:2], 7)==False) and (is_point_in_circle(dict_joints['L_Wrist'][:2], dict_joints["L_Shoulder"][:2], 7)==False) and (is_point_in_circle(dict_joints['L_Wrist'][:2], dict_joints["L_Elbow"][:2], 7)==False):
                dict_joints["L_Wrist"][2]=1
    else:
        if (dict_joints['L_Hip'][0]-dict_joints['L_Knee'][0])!=0:
            _, _, slo1, inter1 = create_line(dict_joints['L_Hip'],dict_joints['L_Knee']) 
            d_l_w, p_L_Wrist = distance_point_line(dict_joints['L_Wrist'], slo1, inter1, threshold=0.0)
        else:
            d_l_w, p_L_Wrist = distance_point_line_noSlope(dict_joints['L_Wrist'], dict_joints['L_Hip'][0], threshold=0.0)
        
        if (p_L_Wrist==0) and (d_l_w>7):
            dict_joints["L_Wrist"][2]=1
    
    # print(dist_l_w, "······Relation about L_Wrist with the linea: ", position_L_Wrist)
    # print(dist_r_w, "······Relation about R_Wrist with the linea: ", position_R_Wrist)   
    return dict_joints

def zona_inferior_cuerpo_IR(dict_joints, position):
    #----------------------R_Knee and L_Knee--------------------------------------------------
    """ pin-point right knee and left knee"""
    d1=calculate_distance(dict_joints["R_Knee"],dict_joints["L_Knee"]) 
    if d1>20: #suponemos que la distancia entre dos rodillas/el threshold=20 pixes
        dict_joints["R_Knee"][2]=1
        dict_joints["L_Knee"][2]=1
    else:
        if position==0: #left side of the line --> es pose right side
            dict_joints["R_Knee"][2]=1
            dict_joints["L_Knee"][2]=0
        elif position==1:
            dict_joints["R_Knee"][2]=0
            dict_joints["L_Knee"][2]=1
        else: ######DOUBT NOT SURE IS CORRECTLY 
            dict_joints["R_Knee"][2]=1
            dict_joints["L_Knee"][2]=1

    d2_1=calculate_distance(dict_joints["R_Knee"],dict_joints["R_Hip"])
    d2_2=calculate_distance(dict_joints["R_Knee"],dict_joints["R_Ankle"])
    # print("R_Kee a R_Hip: ", d2_1)
    # print("R_Kee a R_Ankle: ", d2_2)
    if (d2_1<35) | (d2_2<35):#Bend knees perpendicular to mattress surface (threshold=30 no --> 25)
        dict_joints["R_Knee"][2]=0

    d4_1=calculate_distance(dict_joints["L_Knee"],dict_joints["L_Hip"])
    d4_2=calculate_distance(dict_joints["L_Knee"],dict_joints["L_Ankle"])
    # print("L_Kee a L_Hip: ", d4_1)
    # print("L_Kee a L_Ankle: ", d4_2)
    if (d4_1<35) | (d4_2<35):#Bend knees perpendicular to mattress surface ( threshould=30) 
        dict_joints["L_Knee"][2]=0
    #-------------------------------------------------------------------------------------

    #-----------------------R_Ankle and L_Ankle-------------------------------------------
    """ pinpoint right ankle and left ankle """
    d3=calculate_distance(dict_joints["R_Ankle"],dict_joints["L_Ankle"])
    if d3>20: 
        dict_joints["R_Ankle"][2]=1
        dict_joints["L_Ankle"][2]=1
    else:
        if position==0:
            dict_joints["R_Ankle"][2]=1
            dict_joints["L_Ankle"][2]=0
        elif position==1:
            dict_joints["R_Ankle"][2]=0
            dict_joints["L_Ankle"][2]=1
        else:
            dict_joints["R_Ankle"][2]=1
            dict_joints["L_Ankle"][2]=1
            # dict_joints["R_Ankle"][2]=dict_joints["R_Ankle"][2]
            # dict_joints["L_Ankle"][2]=dict_joints["L_Ankle"][2]
    
    #if r_ankle and l_ankle ==1 check:
    d2=calculate_distance(dict_joints["R_Knee"],dict_joints["L_Ankle"])
    if d2<20:
        if position==0:
            dict_joints["L_Ankle"][2]=0
    d4=calculate_distance(dict_joints["L_Knee"],dict_joints["R_Ankle"])
    if d4<20:
        if position==1:
            dict_joints["R_Ankle"][2]=0

    #if r_ankle and l_ankle ==1 check:
    #check distance between line(r_hip and r_knee) with L_ankle. and line(r_ankle and r_knee) with l_ankle
    xt, yt, slope, intercept = create_line(dict_joints['R_Hip'],dict_joints['R_Knee']) ##when slope, intercept ==0, also need to implement another function
    distance, point_position_L_Elbow = distance_point_line(dict_joints['L_Ankle'], slope, intercept, threshold=0.0)
    # print("diatancia left leg:" ,distance)
    if distance < 16:#
        if position==0:
            dict_joints["L_Ankle"][2]=0

    xt, yt, slope1, intercept1 = create_line(dict_joints['R_Ankle'],dict_joints['R_Knee']) ##when slope, intercept ==0, also need to implement another function
    distance1, point_position_L_Elbow = distance_point_line(dict_joints['L_Ankle'], slope1, intercept1, threshold=0.0)
    if distance1 < 16:
        if position==0:
            dict_joints["L_Ankle"][2]=0


    #check distance between line(l_hip and l_knee) with r_ankle. and line(l_ankle and l_knee) with r_ankle
    xt, yt, slope, intercept = create_line(dict_joints['L_Hip'],dict_joints['L_Knee']) ##when slope, intercept ==0, also need to implement another function
    distance, point_position_L_Elbow = distance_point_line(dict_joints['R_Ankle'], slope, intercept, threshold=0.0)
    # print("diatancia:" ,distance)
    # print("dict_joints[R_Ankle][1]: ",dict_joints["R_Ankle"][1])
    # print("dict_joints[L_Knee][1]: ",dict_joints["L_Knee"][1])
    # print("dict_joints[L_Hip][1]: ",dict_joints["L_Hip"][1])
    if distance < 16:#20
        if position==1:
            dict_joints["R_Ankle"][2]=0
        if position==2 and dict_joints["R_Ankle"][1]<dict_joints["L_Knee"][1] and dict_joints["R_Ankle"][1]>dict_joints["L_Hip"][1]: ##because the axis y plot (255,0)
            dict_joints["R_Ankle"][2]=0
    # print("dict_joints[R_Ankle][1]: ",dict_joints["R_Ankle"][2])
    # print("dict_joints[L_Ankle][1]: ",dict_joints["L_Ankle"][2])
    xt, yt, slope1, intercept1 = create_line(dict_joints['L_Ankle'],dict_joints['L_Knee']) ##when slope, intercept ==0, also need to implement another function
    distance1, point_position_L_Elbow = distance_point_line(dict_joints['R_Ankle'], slope1, intercept1, threshold=0.0)
    if distance1 < 16:
        # if position==1:
        if position==1 and dict_joints["R_Ankle"][1]<dict_joints["L_Knee"][1] and dict_joints["R_Ankle"][1]>dict_joints["L_Ankle"][1]:
            dict_joints["R_Ankle"][2]=0
        if position==2 and dict_joints["R_Ankle"][1]<dict_joints["L_Knee"][1] and dict_joints["R_Ankle"][1]>dict_joints["L_Ankle"][1]:
            dict_joints["R_Ankle"][2]=0
    

    """ ----------------check the part L_CalfMucle and R_CalfMucle----------------"""
    #calculate angle between two lines
    lineA1=[dict_joints['L_Knee'], dict_joints['L_Ankle']]
    lineB1=[dict_joints['L_Knee'], dict_joints['L_Hip']]
    leftAngle=ang(lineA1, lineB1)
    # print("Left angle: ",leftAngle)
    # print("Left angle: ",ang(lineA, lineB))
    lineA2=[dict_joints['R_Knee'], dict_joints['R_Ankle']]
    lineB2=[dict_joints['R_Knee'], dict_joints['R_Hip']]
    rightAngle=ang(lineA2, lineB2)
    # print("Right angle: ",rightAngle)

    # print(dict_joints["R_Ankle"][2], dict_joints["R_Knee"][2], position)
    # print(dict_joints["L_Ankle"][2], dict_joints["L_Knee"][2])
    
    #----check----- 
    if dict_joints["R_Ankle"][2]==1 and dict_joints["R_Knee"][2]==1 and position==2 and rightAngle>166:
        pm1=(dict_joints["R_Ankle"] + dict_joints["R_Knee"]) /2
        pm1[2]=0
        dict_joints["R_CalfMucle"]=pm1.copy()
        dict_joints["R_CalfMucle"][2]=1
        # dict_joints["R_Ankle"][2]=0
        dict_joints["R_Knee"][2]=0

    if dict_joints["L_Ankle"][2]==1 and dict_joints["L_Knee"][2]==1 and position==2 and leftAngle>166:
        pm2=(dict_joints["L_Ankle"] + dict_joints["L_Knee"]) /2
        pm2[2]=0
        dict_joints["L_CalfMucle"]=pm2.copy()
        dict_joints["L_CalfMucle"][2]=1
        # dict_joints["L_Ankle"][2]=0
        dict_joints["L_Knee"][2]=0
   
    return dict_joints

#creating a plot function with the dict_joints (image, axis center, and 14 pinpoints provisional)
def plot_pinpoints(dict_joints, imgvis, output_path=None):
    plt.imshow(imgvis)#print the original image 
   
    xg=[]
    yg=[]
    xr=[]
    yr=[]
    for i in dict_joints.keys():
        # print(type(dict_joints[i]))
        if dict_joints[i][2]==1:
            xr.append(dict_joints[i][0])
            yr.append(dict_joints[i][1])
        else:
            xg.append(dict_joints[i][0])
            yg.append(dict_joints[i][1])
    # plt.scatter(xg, yg, color='green')
    plt.scatter(xr, yr, color='red')

    xt,yt,position=position_prediction(dict_joints)
    # plt.plot(xt, yt, 'royalblue')  # 'r-' for a red line
    
    plt.xlim([0, 255])
    plt.ylim([255, 0])
    plt.show()

    """ print results with text 
    pinpoints = []
    for i in dict_joints.keys():
        if dict_joints[i][2]==1:
            # print(i, ":", dict_joints[i])
            pinpoints.append(i)
    print("Los puntos de presión son: ")
    print(pinpoints[:6])
    print(pinpoints[6:])
    """
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

#plots points of joints with radius=3.5
def plot_Circles(dict_pred, imgvis):
    #plt.imshow(imgvis)#print the original image 
    xt,yt,position=position_prediction(dict_pred)
    plt.plot(xt, yt, 'royalblue')  # 'r-' for a red line
    
    x=[]
    for i in dict_pred.keys():
        x.append(plt.Circle(dict_pred[i][:2],radius=3.5,color="orange")) # #define circles and add to the list x # c1=plt.Circle((5, 5), radius=1)

    for ix in x:
        plt.gca().add_artist(ix) #add circles to plot
        
    plt.xlim([0, 255])
    plt.ylim([255, 0])
    plt.show()


def calculate_distance(p1, p2):
    """ Calculate distance between two pionts"""
    # distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    distance = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
    return distance

def ang(lineA, lineB):
    """ Calculate angle between two lines """
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]

    dot_prod = dot(vA, vB)
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5

    
    cos_ = dot_prod / (magA * magB)
    cos_ = max(min(cos_, 1), -1)  

    angle = math.acos(cos_)
    ang_deg = math.degrees(angle)%360
    
    if ang_deg - 180 >= 0:
        return 360 - ang_deg
    else:
        return ang_deg
def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]


def is_point_in_circle(point, center, radius):
    """ Comprova si un punt està dins d'un cercle donat """
    distance = math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
    return distance <= radius


def plot_aproximacio_posicio_corporal(dict_joints, imgvis, output_path=None):
    plt.imshow(imgvis)
    
    # Dibuixar punts vermells a colzes i genolls
    for nom in ["L_Elbow", "R_Elbow", "L_Knee", "R_Knee"]:
        x, y = dict_joints[nom][:2]
        plt.scatter(x, y, color='red', s=40)

    # Dibuixar línia central del tronc
    p1, p2 = points_axis_center(dict_joints)
    xt, yt, _, _ = create_line(p1, p2)
    plt.plot(xt, yt, color='blue')

    plt.xlim([0, 255])
    plt.ylim([255, 0])  # Invertir eix Y per coincidir amb imatge
    plt.axis('off')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()



def plot_amb_pinpoints_al_costat(img, dict_joints):
    """
    Mostra la imatge amb punts de pressió i una llista al costat amb els noms.
    """
    # Obtenim els noms dels punts amb pressió
    pinpoints = [k for k, v in dict_joints.items() if v[2] == 1]

    # Crear la figura amb dues columnes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 5), gridspec_kw={'width_ratios': [3, 1]})
    
    # Imatge amb punts
    ax1.imshow(img)
    for k in pinpoints:
        x, y = dict_joints[k][:2]
        ax1.plot(x, y, 'ro')
    ax1.axis('off')
    ax1.set_title("Imatge amb punts de pressió")

    # Llista al costat
    ax2.axis('off')
    ax2.set_title("📍 Pin-points detectats:")
    text = "\n".join(pinpoints)
    ax2.text(0.1, 1, text, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.show()

def plot_amb_punts_i_llista(dict_joints, img, title="Punts de pressió detectats"):
    # Punts de pressió
    pinpoints = [(k, v[:2]) for k, v in dict_joints.items()
                 if isinstance(v, (list, np.ndarray)) and len(v) > 2 and v[2] == 1]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # A l'esquerra: la imatge amb els punts
    axs[0].imshow(img, cmap="cool")
    for nom, coord in pinpoints:
        axs[0].scatter(coord[0], coord[1], color='red', s=40)
    axs[0].set_title("Imatge amb punts de pressió", fontsize=12)
    axs[0].axis('off')

    # A la dreta: la llegenda
    noms_formatats = "\n".join([f"• {k.replace('_',' ').capitalize()}" for k, _ in pinpoints])
    axs[1].axis('off')
    axs[1].text(0.05, 0.5, f"{title}:\n\n{noms_formatats}", fontsize=12, va='center')

    plt.tight_layout()
    plt.show()






if __name__ == '__main__':
    

    
    imgtype = 'IR'
    imgpath = r'C:\Users\Asus\Desktop\TFG\prova\pre_231218171115_Decubit Supi_Neutre_Roba_IR.png'
    saveimgname = 'output_skeleton'
    print(" Guardant prediccions....")
    img_ori, pred, _ = predAndPrintImg(imgpath, imgtype)

    save_predictions(pred)
    
    dict_joints = genDict(pred)
    
    detectar_aixecat_lateral(dict_joints, debug= True)
    

    
    xt, yt, position = position_prediction(dict_joints)
    plot_aproximacio_posicio_corporal(dict_joints, img_ori)

    dict_joints = zona_cabezal_IR(dict_joints, position)
    dict_joints = zona_superior_cuerpo_IR(dict_joints, position)
    dict_joints = zona_inferior_cuerpo_IR(dict_joints, position)

    pinpoints = [k for k, v in dict_joints.items() if v[2] == 1]
    print("\n📍 Pin-points detectats:")
    print(pinpoints)

    plot_pinpoints(dict_joints, img_ori)
    



    
    
    
