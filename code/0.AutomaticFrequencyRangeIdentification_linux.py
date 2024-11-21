
def plotrectangle_ans():
    #Plot ans time by rectangle
    ref_tmpname_POS="123.wav"
    ref_tmpname_UNK="456.wav"
    for csvfilename in glob.glob(f'./*/*.csv', recursive=True): #read all wav file in all folder
        #print("Ans csvfilename =",csvfilename)
        path=csvfilename.split("/")[1]
        csvfilename=csvfilename.split("/")[2]
        if audiofilename[:-3]==csvfilename[:-3]: #if name of wav == name of csv, read csv and then plot rectangle
            #print("QQ")
            with open("./%s/%s" %(path,csvfilename), 'r') as reffile: #read csv
                ref_csvreader = csv.DictReader(reffile) ## read rows into a dictionary format
                for row in ref_csvreader: # read a row as {column1: value1, column2: value2,...}
                    ##firstkey == row['Audiofilename']
                    firstkey=list(row.keys())[0] #first key in a dictionary
                    #print("row=",row)             #row= {'嚜澤udiofilename': 'RD_05.wav', 'Starttime': '0', 'Endtime': '1.113281224', 'Q': 'POS'}
                    #print("type(row)=",type(row)) #type(row)= <class 'dict'>
                    #print("firstkey=",firstkey)   #firstkey= 嚜澤udiofilename       
                    #if target class event happened('POS'), plot rectangle
                    if (row['Q']=='POS'):
                        reference_Starttime=float(row['Starttime'])
                        reference_Endtime=float(row['Endtime'])
                        ax = plt.gca()
                        #avoid multi label
                        if (ref_tmpname_POS!=firstkey):
                            rect = patches.Rectangle((reference_Starttime,real_fs/2/140), (reference_Endtime-reference_Starttime), real_fs/2-(10*real_fs/2/160), linewidth=2, edgecolor='white',linestyle=':', label='Ans of POS', fill = False)
                            ax.add_patch(rect)
                        if (ref_tmpname_POS==firstkey):
                            rect = patches.Rectangle((reference_Starttime,real_fs/2/140), (reference_Endtime-reference_Starttime), real_fs/2-(10*real_fs/2/160), linewidth=2, edgecolor='white',linestyle=':', fill = False)
                            ax.add_patch(rect)
                        ref_tmpname_POS=firstkey
                    #UNK indicates uncertainty about a class and participants can choose to ignore it.
                    if (row['Q']=='UNK'):
                        reference_Starttime=float(row['Starttime'])
                        reference_Endtime=float(row['Endtime'])
                        ax = plt.gca()
                        #avoid multi label
                        if (ref_tmpname_UNK!=firstkey):
                            rect = patches.Rectangle((reference_Starttime,real_fs/2/140), (reference_Endtime-reference_Starttime), real_fs/2-(10*real_fs/2/160), linewidth=2, edgecolor='gray',linestyle=':', label='Ans of UNK', fill = False)
                            ax.add_patch(rect)
                        if (ref_tmpname_UNK==firstkey):
                            rect = patches.Rectangle((reference_Starttime,real_fs/2/140), (reference_Endtime-reference_Starttime), real_fs/2-(10*real_fs/2/160), linewidth=2, edgecolor='gray',linestyle=':', fill = False)
                            ax.add_patch(rect)
                        ref_tmpname_UNK=firstkey
    plt.legend(loc='upper right') #label loc
    
def wavonsetoffset(): 
    #Plot ans time by rectangle
    onset_list=[]
    offset_list=[]
    onset_unk_list=[]
    offset_unk_list=[]
    for csvfilename in glob.glob(f'./*/*.csv', recursive=True): #read all wav file in all folder
        #print("Ans csvfilename =",csvfilename)
        path=csvfilename.split("/")[1]
        csvfilename=csvfilename.split("/")[2]
        if audiofilename[:-3]==csvfilename[:-3]: #if name of wav == name of csv, read csv and then plot rectangle
            #print("QQ")
            with open("./%s/%s" %(path,csvfilename), 'r') as reffile: #read csv
                ref_csvreader = csv.DictReader(reffile) ## read rows into a dictionary format
                for row in ref_csvreader: # read a row as {column1: value1, column2: value2,...}
                    firstkey=list(row.keys())[0] #first key in a dictionary
                    #print("row=",row['Audiofilename'][:-3])            
                    if (row['Q']=='POS'):
                        reference_Starttime=float(row['Starttime'])
                        reference_Endtime=float(row['Endtime'])
                        onset_list.append(reference_Starttime)
                        offset_list.append(reference_Endtime)
                    if (row['Q']=='UNK'):
                        reference_unk_Starttime=float(row['Starttime'])
                        reference_unk_Endtime=float(row['Endtime'])
                        onset_unk_list.append(reference_unk_Starttime)
                        offset_unk_list.append(reference_unk_Endtime)
    return onset_list,offset_list,onset_unk_list,offset_unk_list

def cossim(vec1, vec2):
    cos_sim = np.dot(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim
    
def autosearchminbinandmaxbin(maxlocbi,value_thresh,avg_compare_5posneg_tmp,searchLbin_thresh,searchRbin_thresh):
    avg_compare_5posneg_tmp[avg_compare_5posneg_tmp <= value_thresh] = 0 #if < value_thresh, set to 0
    auto_minbi=maxlocbi
    auto_maxbi=maxlocbi
    left_conti=1
    right_conti=1
    while left_conti==1:   #search left side of maxlocbi
        if (auto_minbi-searchLbin_thresh)<0:
            searchLbin_thresh=4
            if (auto_minbi-searchLbin_thresh)<0:
                print("In autosearch, (auto_minbi-searchLbin_thresh)<0")
                left_conti=0
                auto_minbi=2
                break
        if avg_compare_5posneg_tmp[auto_minbi-1]>=value_thresh:
            auto_minbi=auto_minbi-1
        if (avg_compare_5posneg_tmp[auto_minbi-1]<value_thresh) & (all([val == 0 for val in avg_compare_5posneg_tmp[auto_minbi-searchLbin_thresh:auto_minbi]])==True):
            left_conti=0
        if (avg_compare_5posneg_tmp[auto_minbi-1]<value_thresh) & (all([val == 0 for val in avg_compare_5posneg_tmp[auto_minbi-searchLbin_thresh:auto_minbi]])==False):
            auto_minbi=auto_minbi-1
    while right_conti==1:   #search right side of maxlocbi
        if (auto_maxbi+searchRbin_thresh)>128:
            searchRbin_thresh=4
            if (auto_maxbi+searchRbin_thresh)>128:
                print("In autosearch, (auto_maxbi+searchRbin_thresh)>128")
                right_conti=0
                auto_maxbi=125
                break
        if avg_compare_5posneg_tmp[auto_maxbi+1]>=value_thresh:
            auto_maxbi=auto_maxbi+1
        if (avg_compare_5posneg_tmp[auto_maxbi+1]<value_thresh) & (all([val == 0 for val in avg_compare_5posneg_tmp[auto_maxbi:auto_maxbi+searchRbin_thresh]])==True):
            right_conti=0
        if (avg_compare_5posneg_tmp[auto_maxbi+1]<value_thresh) & (all([val == 0 for val in avg_compare_5posneg_tmp[auto_maxbi:auto_maxbi+searchRbin_thresh]])==False):
            auto_maxbi=auto_maxbi+1
    return auto_minbi,auto_maxbi
    
    







#20240304 KAO, SHENG-LUN

#from scipy.datasets import electrocardiogram #
from scipy import signal
from scipy.signal import butter, lfilter, wiener, find_peaks
from collections import defaultdict
from dtaidistance import dtw
import os
import glob
import numpy
import numpy as np
import csv
import copy
import math
import soundfile as sf
import sounddevice as sd
import time
import librosa
import librosa.display #AttributeError: module 'librosa' has no attribute 'display'
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl #OverflowError: In draw_path: Exceeded cell block limit
mpl.rcParams['agg.path.chunksize'] = 1000000 #OverflowError: In draw_path: Exceeded cell block limit
import matplotlib.colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Start measuring execution time
start = time.perf_counter()

title_font = {'size' : '20'} #標題的字大小
axis_font = {'size' : '18'} #XY軸的字大小
tick_font= {'size' : '12'} #XY刻度的字大小

amp2pa = 1
my_dpi=96
x_dpi=20000 # have to < 2^16(65536)
y_dpi=1000

plottime=1 #Plot time by rectangle?  0 -> no, 1 -> yes
bandp=1
tmpaudiofilename=''
nameandsearchwindowlist=[]
nameandautominbinandautomaxbin=[]

for audiofilename in glob.glob(f'./*/*.wav', recursive=True): #read all wav file in all folder
    fivepos=[1,1,1,1,1]  #Apply this shot?   0 -> no, 1 -> yes
    fiveneg=[1,1,1,1,1]  #Apply this shot?   0 -> no, 1 -> yes

    path=audiofilename.split("/")[1]
    audiofilename=audiofilename.split("/")[2]
    print("\naudiofilename=",audiofilename)
    
    realsignal, real_fs = librosa.load(os.path.join("./"+path+"//",audiofilename))
    
    audiotime=len(realsignal)/real_fs
    print("len(realsignal) =",len(realsignal))
    print("real_fs=",real_fs)
    print("realsignal.shape =",realsignal.shape)
    print("time(sec)=",audiotime)

    ##plot Mel-spectrogram_Hz v.s Time
    S1 = librosa.feature.melspectrogram(y=realsignal, sr=real_fs, n_fft=1024, window='hann', n_mels=128, hop_length=256, fmax=real_fs/2, power=2)
    S_dB = librosa.power_to_db(S1, ref=np.max)

    #Normalized Data by MinMaxScaler
    #normalized = (x-min(x))/(max(x)-min(x)) #min=-80, max=0
    S_dB = (S_dB-(-80))/(0-(-80)) 
    
    ## 5 shot check and get onset/offset+duration of POS/NEG
    onsetoffset=wavonsetoffset()
    firstfiveonset=onsetoffset[0][0:5]
    firstfiveoffset=onsetoffset[1][0:5]
    print("\nonsetoffset=",onsetoffset)
    print("firstfiveonset=",firstfiveonset)
    print("firstfiveoffset=",firstfiveoffset)
    unk_onset=onsetoffset[2]
    unk_offset=onsetoffset[3]
    print("unk_onset=",unk_onset)
    print("unk_offset=",unk_offset)
    if len(unk_onset)==0:
        firstfiveunk_onset=[]
        firstfiveunk_offset=[]
        print("Empty UNK")
    if len(unk_onset)>0:
        if unk_onset[0]>firstfiveoffset[-1]:
            firstfiveunk_onset=[]
            firstfiveunk_offset=[]
            print("A")
        if unk_onset[-1]<firstfiveoffset[-1]:
            firstfiveunk_onset=unk_onset
            firstfiveunk_offset=unk_offset
            print("B")
        if (unk_onset[0]<firstfiveoffset[-1]) & (unk_onset[-1]>firstfiveoffset[-1]):
            for i in range(len(unk_onset)):
                if unk_onset[i]>firstfiveoffset[-1]:
                    firstfiveunk_onset=unk_onset[0:i]
                    firstfiveunk_offset=unk_offset[0:i]
                    print("C")
                    break
    print("firstfiveunk_onset=",firstfiveunk_onset)
    print("firstfiveunk_offset=",firstfiveunk_offset)
    
    coeff1= (S_dB.shape[1])/audiotime
    coeff2= audiotime/(S_dB.shape[1])
    print("coeff1=",coeff1)
    print("coeff2=",coeff2)
    posstart=[]
    posend=[]
    negstart=[]
    negend=[]
    durpos=[]
    durneg=[]
    orig_durneg=[]
    for i in range(5):
        #print(i)
        posstart.append(math.floor(firstfiveonset[i]*coeff1))     #rounding down unconditionally
        posend.append(math.ceil(firstfiveoffset[i]*coeff1))   #rounding up unconditionally
        pos_time=posend[i]-posstart[i]
        if i==0:
            negstart.append(0)
            negend.append(posstart[i])
            neg_time=negend[i]-negstart[i]
        if i>0:
            negstart.append(posend[i-1])
            negend.append(posstart[i])
            neg_time=negend[i]-negstart[i]
        durpos.append(pos_time)
        durneg.append(neg_time)
        orig_durneg.append(neg_time)
    print("\nposstart=",posstart)
    print("posend=",posend)
    print("negstart=",negstart)
    print("negend=",negend)
    print("durpos=",durpos)
    print("durneg=",durneg)
    print("orig_durneg=",orig_durneg)
    unkstart=[]
    unkend=[]
    for i in range(len(firstfiveunk_onset)):
        unkstart.append(math.floor(firstfiveunk_onset[i]*coeff1))     #rounding down unconditionally
        unkend.append(math.ceil(firstfiveunk_offset[i]*coeff1))   #rounding up unconditionally
    print("unkstart=",unkstart)
    print("unkend=",unkend) 

    
    ##Define serarch window length and delete short NEG
    #searchwindow=min(durpos)
    if (min(durneg)<min(durpos)) & (durneg[0]==min(durneg)):
        #delete first NEG shot if too short
        fiveneg[0]=0
        #del durneg[0]
    if (max(durpos) <=250):
        tmplen=min(durpos)
        searchwindow=tmplen
        zeroloc=[]
        for i in range (len(fiveneg)):
            if durneg[i]<(tmplen-(tmplen*0.12)):
                fiveneg[i]=0
        for i in range(len(fiveneg)):
            if fiveneg[i]==0:
                zeroloc.append(i)
        for i in sorted(zeroloc, reverse=True):
            del durneg[i]
        if min(durneg)<tmplen:
            searchwindow=min(durneg)
    if (max(durpos) >250) & (max(durpos)<=450 ):
        tmplen=int((max(durpos))/3)
        if min(durpos)<tmplen:
            tmplen=min(durpos)
        for i in range (len(fiveneg)):
            if durneg[i]<(tmplen):
                fiveneg[i]=0
        searchwindow=tmplen
    if (max(durpos) >450) & (max(durpos)<=900 ):
        tmplen=int((max(durpos))/5.5)
        if min(durpos)<tmplen:
            tmplen=min(durpos)
        for i in range (len(fiveneg)):
            if durneg[i]<(tmplen):
                fiveneg[i]=0
        searchwindow=tmplen
    if (max(durpos) >900):
        if fiveneg[0]==0:
            del durneg[0]
        if min(durneg)>=min(durpos):
            searchwindow=min(durpos)
        if min(durneg)<min(durpos):
            searchwindow=min(durneg)
    print("max(durpos)=",max(durpos)) 
    print("max(durneg)=",max(durneg)) 
    print("min(durpos)=",min(durpos)) 
    print("min(durneg)=",min(durneg)) 
    print("searchwindow length=",searchwindow)
    print("audiofilename=",audiofilename)
    print("fivepos=",fivepos)
    print("fiveneg=",fiveneg)
    nameandsearchwindowlist.append(audiofilename)
    nameandsearchwindowlist.append(searchwindow)
    nameandsearchwindowlist.append("|||")
    
    
    print("\nSelect NEG")
    for i in range(5):
        #print("\ni=",i)
        if i ==0:
            count_unk=0
        if i ==1:
            count_unk=0
        if i ==2:
            count_unk=0
        if i ==3:
            count_unk=0
        if i ==4:
            count_unk=0
        #UNK doesn't trigger on NEG events.
        for un in range(len(unkstart)):
            if (unkstart[un]>=negstart[i]-1) & (unkstart[un]<=negend[i]):
                count_unk=count_unk+1
        print("Check if UNK in NEG, %d NEG shot, count_UNK=%d" %(i+1,count_unk))
        #ignore this NEG shot, if >= 1 UNK
        if count_unk>=1:
            fiveneg[i]=0
            print(" >=1 UNK in %d NEG shot" %(i+1))
    print("fiveneg=",fiveneg)

    
    ##Step1: Auto obtain frequency range in first 5 POS shot V4
    print("\n\nAuto obtain frequency range in first 5 POS shot V4")
    debris=10
    splitnumb=(math.ceil(128/debris)) #13 segments
    print("splitnumb=",splitnumb)
    compare_5posneg=[]
    minposduration=min(durpos)
    print("minposduration=",minposduration)
    for a in range(5):  #5 POS shot
        print("\n%d POS shot" %(a+1))
        if fivepos[a]==1:
            xnum=math.ceil(durpos[a]/minposduration)   #rounding up unconditionally
            print("xnum=",xnum)
            #pos_S_dB=S_dB[:,posstart[a]:posend[a]]
            posduration=durpos[a]
            appliableneg=[]
            #find appliable neg in this pos shot.
            for j in range(5):
                #print("j=",j)
                negduration=orig_durneg[j]
                if (minposduration>negduration) | (fiveneg[j]==0):
                    #print("QQ ij=",a,j)
                    appliableneg.append(0)
                else:
                    appliableneg.append(1)
            print("appliableneg=",appliableneg) 

            if all([val == 0 for val in appliableneg])==True:
                print("appliableneg all zero in %d POS shot" %(a+1))
                continue
    
            #split frequency range by splitnumb
            for h in range(splitnumb):
                CS_widthheight_POSandNEG_five=[]
                height_POSandNEG_five=[]
                #print("h=",h)
                st=h*debris
                en=st+debris
                if h==splitnumb-1:
                    en=128
                print("\nst=",st)
                print("en=",en)
                #compare_5posneg=[]
                split_S_dB=S_dB[st:en,:]
                pos_S_dB=S_dB[:,posstart[a]:posend[a]]
                pos_S_dB=pos_S_dB[st:en,:]
                print("split_S_dB.shape=",split_S_dB.shape)
                print("pos_S_dB.shape=",pos_S_dB.shape)
                
                #split time duration by minposduration
                tmp_cossim_pos_width_melspe=[]
                tmp_cossim_pos_height_melspe=[]
                xnum_possubstractneg_height=[]
                print("pos_S_dB.shape[1]-minposduration+1=",pos_S_dB.shape[1]-minposduration+1)
                for k in range (xnum):
                    #print("k=",k)
                    stt=k*minposduration
                    enn=stt+minposduration
                    if k==xnum-1:
                        print(" k==xnum-1 in %d POS" %(a+1))
                        stt=durpos[a]-minposduration
                        enn=durpos[a]
                    print("stt=%d in %d POS" %(stt,a+1))
                    print("enn=%d in %d POS" %(enn,a+1))
                    tmp_pos_S_dB=pos_S_dB[:,stt:enn]
                    tmp_pos_width_melspe=tmp_pos_S_dB.sum(axis=0) / tmp_pos_S_dB.shape[0]
                    tmp_pos_height_melspe=tmp_pos_S_dB.sum(axis=1) / tmp_pos_S_dB.shape[1]
                    for b in range(5): #search 5 NEG shot in each POS shot
                        if appliableneg[b]==0: #ignore this NEG shot
                            if a==0:
                                print("Ignore this NEG shot b=",b)
                            continue
                        if appliableneg[b]==1: 
                            CS_widthheight_POSandNEG_one=[]
                            height_POSandNEG_one=[]
                            #print("b=",b)
                            tmp_pos_S_dB_width=tmp_pos_S_dB.sum(axis=0) / tmp_pos_S_dB.shape[0]
                            tmp_pos_S_dB_height=tmp_pos_S_dB.sum(axis=1) / tmp_pos_S_dB.shape[1]
                            neg_S_dB=split_S_dB[:,negstart[b]:negend[b]]
                            #print("neg_S_dB=",neg_S_dB)
                            #print("neg_S_dB.shape=",neg_S_dB.shape)
                            if neg_S_dB.shape[1] >= tmp_pos_S_dB.shape[1]:
                                #similar POS in NEG
                                #print("neg_S_dB.shape[1] > tmp_pos_S_dB.shape[1]")
                                tmp_cossim_neg_widthheight_melspe=[]
                                tmp_height_melspe=[]
                                #print("cc=",neg_S_dB.shape[1]-tmp_pos_S_dB.shape[1]+1)
                                for c in range (neg_S_dB.shape[1]-tmp_pos_S_dB.shape[1]+1):
                                    stt=c
                                    enn=c+tmp_pos_S_dB.shape[1]

                                    tmp_neg_S_dB=neg_S_dB[:,stt:enn]
                                    tmp_neg_width_melspe=tmp_neg_S_dB.sum(axis=0) / tmp_neg_S_dB.shape[0]
                                    tmp_neg_height_melspe=tmp_neg_S_dB.sum(axis=1) / tmp_neg_S_dB.shape[1]
                                    #print("tmp_neg_height_melspe.shape=",tmp_neg_height_melspe.shape)
                                    #print("tmp_pos_S_dB_height.shape=",tmp_pos_S_dB_height.shape)
                                    tmp_cossim_neg_widthheight_melspe.append((cossim(tmp_neg_width_melspe,tmp_pos_width_melspe)+cossim(tmp_neg_height_melspe,tmp_pos_height_melspe))/2)
                                    tmp_height_melspe.append(tmp_neg_S_dB.sum(axis=1))
                                CS_widthheight_POSandNEG_one.append(tmp_cossim_neg_widthheight_melspe)
                                height_POSandNEG_one.append(tmp_height_melspe)
                            CS_widthheight_POSandNEG_five.append(CS_widthheight_POSandNEG_one)
                            height_POSandNEG_five.append(height_POSandNEG_one)
                    print("len(CS_widthheight_POSandNEG_five)=",len(CS_widthheight_POSandNEG_five))
                    print("len(height_POSandNEG_five)=",len(height_POSandNEG_five))

                    tmplist=[]
                    tmpheight=[]
                    for d in range(len(CS_widthheight_POSandNEG_five)):
                        tmplist=tmplist+CS_widthheight_POSandNEG_five[d][0]
                        tmpheight=tmpheight+height_POSandNEG_five[d][0]
                        #print("len(CS_widthheight_POSandNEG_five[d])=",len(CS_widthheight_POSandNEG_five[d]))
                        #print("len(tmpheight[d])=",len(tmpheight[d]))
                    loc_max=tmplist.index(max(tmplist))  #searching most similar POS in NEG
                    print("\nlen(tmplist)=",len(tmplist))
                    tmp_possubstractneg_height=tmp_pos_S_dB.sum(axis=1)-tmpheight[loc_max]  #POS-NEG
                    #possubstractneg_height=tmp_pos_S_dB.sum(axis=1)-neg_S_dB.sum(axis=1)
                    print("tmp_possubstractneg_height.shape=",tmp_possubstractneg_height.shape) 
                    xnum_possubstractneg_height.append(tmp_possubstractneg_height)
                    if k==xnum-1:
                        print("xnum_possubstractneg_height=",xnum_possubstractneg_height)
                        print("len(xnum_possubstractneg_height)=",len(xnum_possubstractneg_height)) 
                        avg_xnum_possubstractneg_height=np.mean(xnum_possubstractneg_height,axis=0)
                        #med_xnum_possubstractneg_height=np.median(xnum_possubstractneg_height,axis=0)
                        possubstractneg_height=avg_xnum_possubstractneg_height
                        #possubstractneg_height=med_xnum_possubstractneg_height
                        if h==0:
                            all_possubstractneg_height=possubstractneg_height
                        if h>0:
                            all_possubstractneg_height=np.concatenate((all_possubstractneg_height, possubstractneg_height))
                        if h==splitnumb-1: #plot fig
                            print("h==splitnumb-1")
                            all_possubstractneg_height[all_possubstractneg_height < 0] = 0 #if array <0, array=0
                            #Normalized Data by MinMaxScaler
                            #normalized = (x-min(x))/(max(x)-min(x)) #min=-80, max=0
                            all_possubstractneg_height = (all_possubstractneg_height-min(all_possubstractneg_height))/(max(all_possubstractneg_height)-min(all_possubstractneg_height))
                            print("\nAll 'nan' in all_possubstractneg_height ? = %s in %d POS shot" %(np.isnan(all_possubstractneg_height).all(),a+1))

                            if np.isnan(all_possubstractneg_height).all() == False: #run below if all_possubstractneg_height != nan
                                compare_5posneg.append(all_possubstractneg_height)
                                print("tmp_pos_S_dB.shape=",tmp_pos_S_dB.shape) 
                                print("neg_S_dB.shape=",neg_S_dB.shape) 
                                print("all_possubstractneg_height.shape=",all_possubstractneg_height.shape)
                                peaks, _ = find_peaks(all_possubstractneg_height, height=0.339)
                                print("bin peaks=",peaks)
                                plt.figure(figsize=(5, 5), dpi=my_dpi) # pixel image
                                plt.plot(all_possubstractneg_height)
                                plt.plot(peaks, all_possubstractneg_height[peaks], "x")
                                #plt.plot(all_possubstractneg_height)
                                plt.xlabel('Mel bands (0~127)', **axis_font)
                                peaklist=[]
                                for i in range(len(peaks)):
                                    #print(a[i])
                                    peaklist.append(peaks[i])
                                print("peaklist=",peaklist)
                                plt.savefig('%s_V4_debris=%d_splitnumb=%d_%dshot_binpeaks=%s.png' %(audiofilename,debris,splitnumb,a+1,peaklist))
                                plt.show()
            #print("compare_5posneg=",compare_5posneg) 
            print("\nlen(compare_5posneg)=",len(compare_5posneg))
            print("len(compare_5posneg[0])=",len(compare_5posneg[0]))
    
    
    #AVERAGE 5 POS-NEG
    avg_compare_5posneg=np.average(compare_5posneg, axis=0)
    avg_compare_5posneg_orig = avg_compare_5posneg.copy()
    print("avg_compare_5posneg=",avg_compare_5posneg)
    print("len avg_compare_5posneg=",len(avg_compare_5posneg))
    print("avg_compare_5posneg_orig=",avg_compare_5posneg_orig)
    print("len avg_compare_5posneg_orig=",len(avg_compare_5posneg_orig))
    print("\nAVERAGE 5 POS-NEG")
    print("len(avg_compare_5posneg)=",len(avg_compare_5posneg)) #128
    print("type(avg_compare_5posneg)=",type(avg_compare_5posneg)) #<class 'numpy.ndarray'>
    print("avg_compare_5posneg.shape=",avg_compare_5posneg.shape) #(128,)
    avg_peaks, _ = find_peaks(avg_compare_5posneg, height=0.339)
    print("bin avg_peaks=",avg_peaks)
    plt.figure(figsize=(5, 5), dpi=my_dpi) # pixel image
    plt.plot(avg_compare_5posneg)
    plt.plot(avg_peaks, avg_compare_5posneg[avg_peaks], "x")
    #plt.plot(avg_compare_5posneg)
    plt.xticks( **tick_font)
    plt.yticks( **tick_font)
    plt.xlabel('Mel bands (0~127)', **axis_font)
    avg_peaklist=[]
    for i in range(len(avg_peaks)):
        #print(a[i])
        avg_peaklist.append(avg_peaks[i])
    print("orig avg_peaklist=",avg_peaklist)
    print("orig len avg_peaklist=",len(avg_peaklist))
    plt.savefig('%s_V4_debris=%d_splitnumb=%d_AVERAGEshot_binpeaks=%s.png' %(audiofilename,debris,splitnumb,avg_peaklist))
    plt.show()
    
    #MEDIAN 5 POS-NEG
    print("\nMEDIAN 5 POS-NEG")
    med_compare_5posneg=np.median(compare_5posneg, axis=0)
    print("len(med_compare_5posneg)=",len(med_compare_5posneg))  
    med_peaks, _ = find_peaks(med_compare_5posneg, height=0.339)
    print("bin med_peaks=",med_peaks)
    plt.figure(figsize=(5, 5), dpi=my_dpi) # pixel image
    plt.plot(med_compare_5posneg)
    plt.plot(med_peaks, med_compare_5posneg[med_peaks], "x")
    #plt.plot(med_compare_5posneg)
    plt.xticks( **tick_font)
    plt.yticks( **tick_font)
    #plt.xlabel('Frequency bins (0~127)', **axis_font)
    plt.xlabel('Mel bands (0~127)', **axis_font)
    med_peaklist=[]
    for i in range(len(med_peaks)):
        #print(a[i])
        med_peaklist.append(med_peaks[i])
    print("med_peaklist=",med_peaklist)
    plt.savefig('%s_V4_debris=%d_splitnumb=%d_MEDIANshot_binpeaks=%s.png' %(audiofilename,debris,splitnumb,med_peaklist))
    plt.show()

    
    #continue
    
    ##Step2: use AVERAGE 5 POS-NEG to find auto_minbin + auto_maxbin.
    tt=0
    value_threshold=0.23
    searchLbin_threshold=7
    searchRbin_threshold=7
    maxlocbin=np.argmax(avg_compare_5posneg)
    searchresult=autosearchminbinandmaxbin(maxlocbin,value_threshold,avg_compare_5posneg,searchLbin_threshold,searchRbin_threshold)
    #print("searchresult=",searchresult) #(10, 40)
    auto_minbin=searchresult[0]
    auto_maxbin=searchresult[1]
    auto_maxminbin=auto_maxbin-auto_minbin
    print("\n1st value_threshold=%.2f, audiofilename=%s" %(value_threshold,audiofilename))
    print("1st maxlocbin=",maxlocbin)         #Max  bin
    print("1st auto_maxminbin=",auto_maxminbin)
    print("1st auto_minbin=",auto_minbin)     #low  bin
    print("1st auto_maxbin=",auto_maxbin)     #high bin

    if (maxlocbin<=5): #Run again by value_threshold.
        print("\nmaxlocbin<=5. audiofilename=",audiofilename)
        avg_compare_5posneg=avg_compare_5posneg_orig.copy()  #reset
        tt=1
        #ignore values near border(0, 127)
        #avg_compare_5posneg<=7 set to 0
        #avg_compare_5posneg>=120 set to 0
        print("ignore (i<=5) | (i>=120)  in avg_compare_5posneg")
        for i in range(len(avg_compare_5posneg)):
            if (i<=5) | (i>=120):
                avg_compare_5posneg[i]=0
        value_threshold=0.205
        print("Run again in maxlocbin<=5, by value_threshold =",value_threshold)

        maxlocbin=np.argmax(avg_compare_5posneg)
        searchresult=autosearchminbinandmaxbin(maxlocbin,value_threshold,avg_compare_5posneg,searchLbin_threshold,searchRbin_threshold)
        auto_minbin=searchresult[0]
        auto_maxbin=searchresult[1]
        auto_maxminbin=auto_maxbin-auto_minbin
        print("In maxlocbin<=5, Run again auto_maxminbin=%d, audiofilename=%s" %(auto_maxminbin,audiofilename))

    if (auto_maxminbin<=5) | ((maxlocbin<=12) & (auto_maxbin-maxlocbin<=5)):
        print("\n(auto_maxminbin<=5) | (maxlocbin<=12) & (auto_maxbin-maxlocbin<=5). audiofilename=",audiofilename)
        avg_compare_5posneg=avg_compare_5posneg_orig.copy()  #reset
        tt=1
        value_threshold=0.13
        #ignore values near border(0, 127)
        #avg_compare_5posneg<=7 set to 0
        #avg_compare_5posneg>=120 set to 0
        print("ignore (i<=7) | (i>=120)  in avg_compare_5posneg")
        for i in range(len(avg_compare_5posneg)):
            if (i<=7) | (i>=120):
                avg_compare_5posneg[i]=0
        print("Run again in auto_maxminbin<=5, by value_threshold =",value_threshold)

        maxlocbin=np.argmax(avg_compare_5posneg)
        print("maxlocbin=",maxlocbin)
        searchresult=autosearchminbinandmaxbin(maxlocbin,value_threshold,avg_compare_5posneg,searchLbin_threshold,searchRbin_threshold)
        auto_minbin=searchresult[0]
        auto_maxbin=searchresult[1]
        auto_maxminbin=auto_maxbin-auto_minbin
        print("In auto_maxminbin<=5, Run again auto_maxminbin= %d, audiofilename=%s" %(auto_maxminbin,audiofilename))

        if auto_maxminbin<=10: #make auto_maxminbin longer
            print(" Run again auto_maxminbin<=10, audiofilename=%s" %(audiofilename))
            avg_compare_5posneg=avg_compare_5posneg_orig.copy()  #reset
            print("ignore (i<=7) | (i>=120)  in avg_compare_5posneg")
            for i in range(len(avg_compare_5posneg)):
                if (i<=7) | (i>=120):
                    avg_compare_5posneg[i]=0
            value_threshold=0.07
            print(" Run again auto_maxminbin<=10, by value_threshold =",value_threshold)
            avg_compare_5posneg[avg_compare_5posneg <= value_threshold] = 0 #if < value_threshold, set to 0

            avg_peaks, _ = find_peaks(avg_compare_5posneg, height=value_threshold)
            avg_peaklist=[]
            for i in range(len(avg_peaks)):
                #print(a[i])
                avg_peaklist.append(avg_peaks[i])
            print("avg_peaklist=",avg_peaklist)
            avg_compare_5posneg_list=avg_compare_5posneg[avg_peaks].tolist()
            n = 2 #2 biggest values
            result = [avg_compare_5posneg_list.index(i) for i in sorted(avg_compare_5posneg_list, reverse=True)][:n]
            print("result=",result)
            print("len(avg_compare_5posneg)=",len(avg_compare_5posneg))
            little=avg_peaklist[result[0]]
            big=avg_peaklist[result[1]]
            print("little=",little)
            print("big=",big)
            tmpl=[]
            if little==big: #encounter 1st large value == 2nd large value
                print("little==big. audiofilename=",audiofilename)
                for i in range(len(avg_compare_5posneg.tolist())):
                    if avg_compare_5posneg.tolist()[i]==avg_compare_5posneg.tolist()[avg_peaklist[result[0]]]:
                        tmpl.append(i)
                        print(i)
                print("tmpl=",tmpl)
                little=tmpl[0]
                big=tmpl[len(tmpl)-1]
                print("little=",little)
                print("big=",big)
            if little>big:
                little, big = big, little
            for i in range(len(avg_compare_5posneg)): #maxpeak~sencondmaxpeak set to 1
                if (i>=little) & (i<=big):
                    avg_compare_5posneg[i]=1
            print("QQ2 avg_compare_5posneg=",avg_compare_5posneg)

            maxlocbin=np.argmax(avg_compare_5posneg)
            print("maxlocbin=",maxlocbin)
            searchresult=autosearchminbinandmaxbin(maxlocbin,value_threshold,avg_compare_5posneg,searchLbin_threshold,searchRbin_threshold)
            auto_minbin=searchresult[0]
            auto_maxbin=searchresult[1]
            auto_maxminbin=auto_maxbin-auto_minbin
            print("In auto_maxminbin<=5 & auto_maxminbin<=10, Run again auto_maxminbin=%d, audiofilename=%s" %(auto_maxminbin,audiofilename))

    if ((auto_maxminbin >= 59) & (auto_maxminbin <= 62)) & (tt==0): #Run again by value_threshold.
        print("\nauto_maxminbin = 59 ~ 62. audiofilename=",audiofilename)
        avg_compare_5posneg=avg_compare_5posneg_orig.copy() #reset
        tt=1
        value_threshold=0.3118
        print("Run again in auto_maxminbin = 59 ~ 62, by value_threshold =",value_threshold)
        maxlocbin=np.argmax(avg_compare_5posneg)
        searchresult=autosearchminbinandmaxbin(maxlocbin,value_threshold,avg_compare_5posneg,searchLbin_threshold,searchRbin_threshold)
        auto_minbin=searchresult[0]
        auto_maxbin=searchresult[1]
        auto_maxminbin=auto_maxbin-auto_minbin
        print("In auto_maxminbin = 59 ~ 62, Run again auto_maxminbin=%d, audiofilename=%s" %(auto_maxminbin,audiofilename))
    if (auto_maxminbin >= 63) & (tt==0): #Run again by value_threshold.
        print("\nauto_maxminbin >= 63. audiofilename=",audiofilename)
        avg_compare_5posneg=avg_compare_5posneg_orig.copy() #reset
        value_threshold=0.444
        print("Run again in auto_maxminbin >= 63, by value_threshold =",value_threshold)
        maxlocbin=np.argmax(avg_compare_5posneg)
        searchresult=autosearchminbinandmaxbin(maxlocbin,value_threshold,avg_compare_5posneg,searchLbin_threshold,searchRbin_threshold)
        auto_minbin=searchresult[0]
        auto_maxbin=searchresult[1]
        auto_maxminbin=auto_maxbin-auto_minbin
        print("In auto_maxminbin >= 63, Run again auto_maxminbin=%d, audiofilename=%s" %(auto_maxminbin,audiofilename))

    print("\nFinal selectfreqbin, before binsexpand. audiofilename=",audiofilename)
    print("Final maxlocbin=",maxlocbin)         #Max  bin
    print("Final auto_maxminbin=",auto_maxminbin)
    print("Final auto_minbin=",auto_minbin)     #low  bin
    print("Final auto_maxbin=",auto_maxbin)     #high bin
    #binsexpand
    if auto_maxminbin<=20:
        print("  auto_maxminbin<=20. audiofilename=",audiofilename)
        auto_minbin=auto_minbin-3
        auto_maxbin=auto_maxbin+3
    if auto_maxminbin>20:
        print("  auto_maxminbin>20. audiofilename=",audiofilename)
        auto_minbin=auto_minbin-2
        auto_maxbin=auto_maxbin+6
    if auto_minbin<=2:
        print("CC, auto_minbin<=2. audiofilename=",audiofilename)
        auto_minbin=2
    if auto_maxbin>=125:
        print("DD, auto_maxbin>=125. audiofilename=",audiofilename)
        auto_maxbin=125
    print("\nFFinal selectfreqbin, after binsexpand. audiofilename=",audiofilename)
    print("FFinal maxlocbin=",maxlocbin)         #Max  bin
    print("FFinal auto_maxminbin=",auto_maxbin-auto_minbin)
    print("FFinal auto_minbin=",auto_minbin)     #low  bin
    print("FFinal auto_maxbin=",auto_maxbin)     #high bin

    #128 Mel bins in Mel-spectrogram -> frequency(Hz)
    freqs = librosa.core.mel_frequencies(fmin=0, fmax=real_fs/2, n_mels=128)
    auto_minfreq=freqs[auto_minbin]
    auto_maxfreq=freqs[auto_maxbin]
    print("auto_minfreq=",auto_minfreq)   #low  freq
    print("auto_maxfreq=",auto_maxfreq)   #high freq
    

    ##plot Mel-spectrogram_Hz v.s Time
    S1 = librosa.feature.melspectrogram(y=realsignal, sr=real_fs, n_fft=1024, window='hann', n_mels=128, hop_length=256, fmax=real_fs/2, power=2)
    S_dB = librosa.power_to_db(S1, ref=np.max)
    fig, ax = plt.subplots(figsize=(x_dpi/my_dpi, y_dpi/my_dpi), dpi=my_dpi)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=real_fs, fmax=real_fs/2, hop_length=256, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    print("\nS_dB.shape=",S_dB.shape) #(128, 26095)
    #Plot time by rectangle
    if plottime ==1:
        plotrectangle_ans()                                                                       #ans of official
    #labeled freq
    rect = patches.Rectangle((0.5,auto_minfreq), (audiotime-0.5-0.5), auto_maxfreq-auto_minfreq, linewidth=1.3, edgecolor='red',linestyle='-', fill = False)
    ax.add_patch(rect)
    plt.title('Mel-spectrogram', **title_font)
    plt.xticks( **tick_font)
    plt.yticks( **tick_font)
    plt.xlabel('Time(s)', **axis_font)
    plt.ylabel('Frequency(Hz)', **axis_font)
    plt.xlim(0, audiotime)
    plt.savefig('%s_Mel-spectrogram_Auto-freq_autominbin=%d_automaxbin=%d_bandp=%d.png' %(audiofilename,auto_minbin,auto_maxbin,bandp))
    #plt.show()
    if len(realsignal)/real_fs > 120: #if audio longer than 2 minutes(120sec), split it to x times by 2 minutes.
        print(">120sec, Mel-spectrogram")
        x=math.ceil(len(realsignal)/real_fs/120) #x=loop times
        print("x=",x)
        for i in range(x):
            #if i>=3:
            #    break
            plt.xlim(i*120, (i+1)*120) # Set the range of x-axis
            plt.savefig('%s_Mel-spectrogram_Auto-freq_autominbin=%d_automaxbin=%d_bandp=%d, %d-%d.png' %(audiofilename,auto_minbin,auto_maxbin,bandp,i+1,x))
    
    nameandautominbinandautomaxbin.append(audiofilename)
    nameandautominbinandautomaxbin.append(auto_minbin)  #low  bin
    nameandautominbinandautomaxbin.append(auto_maxbin)  #high bin
    nameandautominbinandautomaxbin.append("|||")
    
print("\nnameandsearchwindowlist=",nameandsearchwindowlist)
print("V4nameandautominbinandautomaxbin=",nameandautominbinandautomaxbin)

# Stop measuring execution time
end = time.perf_counter()
executiontime=end-start
print("\nExecution time: %f sec" % (executiontime))
print("Execution time: %f min" % (executiontime/60))
print("Execution time: %f hour" % (executiontime/60/60))
