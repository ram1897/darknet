"""
Darknet Detection Models Training Visualization
Requirements:
1) python3
2) bokeh
Usage:
python thisfilename.py Argument_1 Argument_2 Argument_3
Argument_1: Address of the cluster m/c you are running on (e.g: 192.168.113.203)
Argument_2: Path of the Yolo Config file the training is carried out
Argument_3: The Log directory given at the time of starting Yolo Training
"""

from bokeh.server.server import Server
import csv
import sys
import numpy as np
from bokeh.models import HoverTool, FixedTicker, BoxZoomTool, ResetTool,SaveTool
import re
from bokeh.plotting import figure
from tornado.ioloop import IOLoop

def num_of_lines_in_file(filename):
	#Quicker way to get number of lines than reading the file to a list, and using len(list)
    file = open(filename)
    count = 0
    lines = 0
    buf_size = 1024 * 1024
    read_f = file.read
    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)
    return lines

def modify_doc(doc):

	print("Someone is viewing the Graph")
	
	with open(sys.argv[3]+'lossdump.csv', 'r') as f:
	  reader = csv.reader(f)
	  array = np.array(list(reader))

	iterationarray = (array[:,0]).tolist()
	lossarray = (array[:,1]).tolist()
	learningratearray = (array[:,2]).tolist()
	imgprocessarray = (array[:,3]).tolist()

	for i in range(0,len(iterationarray)):
	    iterationarray[i] = int(float(iterationarray[i].split(':')[1]))
	    lossarray[i] = float(lossarray[i].split(':')[1])
	    learningratearray[i] = float(learningratearray[i].split(':')[1])
	    imgprocessarray[i] = int(imgprocessarray[i].split(':')[1])
	
	hover = HoverTool(
	tooltips=[
	("index", "$index"),
	("(x,y)", "(@x,@y)"),
	]
	)
	# tck = interpolate.UnivariateSpline(imgprocessarray, lossarray)
	# xnew = np.linspace(min(lossarray), max(lossarray), num=len(lossarray), endpoint=True)
	# check = tck(xnew)
	
	# print(type(check))
	
	# print(check.shape)
	fig1 = figure(title='Training Loss Plot',width=1500, height=400,x_axis_label='Images Processed',y_axis_label='Training Loss',tools=[BoxZoomTool(), ResetTool(),hover,SaveTool()])
	fig1.line(x=imgprocessarray, y=lossarray,line_width=1)
	# fig1.line(x=imgprocessarray, y=check.tolist(),line_width=1,line_color="#f46d43" )
	fig1.xaxis[0].formatter.use_scientific = False
	doc.add_root(fig1)

	fig2 = figure(title='Learning Rate Plot',width=1500, height=400,x_axis_label='Images Processed',y_axis_label='Learning Rate Loss',tools=[BoxZoomTool(), ResetTool(),hover,SaveTool()])
	fig2.line(x=imgprocessarray, y=learningratearray,line_width=1 )
	fig2.xaxis[0].formatter.use_scientific = False
	doc.add_root(fig2)

	cfg_file = sys.argv[2]
	with open(sys.argv[2]) as file:
		losslines = file.read().splitlines()

	subdivision = 0
	if ("[yolo]") in losslines:
		num_of_yolo_layers = losslines.count("[yolo]")
	else:
		print("Change the Detection Layer from YOLO to REGION/DETECTION")
		print("Classification Models Visualization not supported now")
		exit(1)

	list2d =  []
	AvgIOU,Class,Obj,NoObj,FIFTYR,SEVENFIVER = [[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]
	count = 0
	linenumber = 0
	tempavgiou,tempclass,tempobj,tempnoobj,temp50R,temp75R = 0.0,0.0,0.0,0.0,0.0,0.0

	for index in range(0,len(losslines)):
		if losslines[index].find("subdivision") !=-1 and losslines[index].find("#") ==-1:
			subdivision = losslines[index]
			subdivision = int(re.findall(r'\b\d+\b', subdivision)[0])

	if subdivision == 0:
		print("Check the subdivisions of your CFG File")
		exit(1)

	with open(sys.argv[3]+"detectiondump.csv") as f:
	    lines = f.readlines()

	linenumber = 0
	batchcount = 0
	counter = 0
	tempavgiou,tempclass,tempobj,tempnoobj,temp50R,temp75R = [],[],[],[],[],[]
	AvgIOU,Class,Obj,NoObj,FIFTYR,SEVENFIVER = [],[],[],[],[],[]
	num_lines = len(lines)


	for index in range(0,num_of_yolo_layers):
		tempavgiou.append([0.0])
		tempclass.append([0.0])
		tempobj.append([0.0])
		tempnoobj.append([0.0])
		temp50R.append([0.0])
		temp75R.append([0.0])
		AvgIOU.append([])
		Class.append([])
		Obj.append([])
		NoObj.append([])
		FIFTYR.append([])
		SEVENFIVER.append([])

	while linenumber < num_lines - 1:
		for index in range(0,num_of_yolo_layers):
			currentline = lines[linenumber].strip()
			if currentline.find("Images_Processed") != -1:
				for index_inner in range(0,num_of_yolo_layers):
					AvgIOU[index_inner].append(tempavgiou[index_inner][0] * 2 /batchcount)
					Class[index_inner].append(tempclass[index_inner][0] * 2 /batchcount)
					Obj[index_inner].append(tempobj[index_inner][0] * 2 /batchcount)
					NoObj[index_inner].append(tempnoobj[index_inner][0] * 2 /batchcount)
					FIFTYR[index_inner].append(temp50R[index_inner][0] * 2 /batchcount)
					SEVENFIVER[index_inner].append(temp75R[index_inner][0] * 2 /batchcount)
				linenumber += 1
				batchcount = 0
				
				tempavgiou,tempclass,tempobj,tempnoobj,temp50R,temp75R = [],[],[],[],[],[]
				for index in range(0,num_of_yolo_layers):
					tempavgiou.append([0.0])
					tempclass.append([0.0])
					tempobj.append([0.0])
					tempnoobj.append([0.0])
					temp50R.append([0.0])
					temp75R.append([0.0])
			else:		
				if currentline.split(',')[1].split(':')[1] != "-nan":
					tempavgiou[index][0] = tempavgiou[index][0] + float(currentline.split(',')[1].split(':')[1])
					tempclass[index][0] = tempclass[index][0] + float(currentline.split(',')[2].split(':')[1])
					tempobj[index][0] = tempobj[index][0] + float(currentline.split(',')[3].split(':')[1])
					tempnoobj[index][0] = tempnoobj[index][0] + float(currentline.split(',')[4].split(':')[1])
					temp50R[index][0] = temp50R[index][0] + float(currentline.split(',')[5].split(':')[1])
					temp75R[index][0] = temp75R[index][0] + float(currentline.split(',')[6].split(':')[1])
				linenumber += 1
				batchcount += 1

	for index in range(0,num_of_yolo_layers):
		fig3 = figure(title='IOU Yolo Layer '+str(index+1),width=1500, height=400,x_axis_label='Images Processed',y_axis_label='IOU of Yolo Layer '+str(index+1),tools=[BoxZoomTool(), ResetTool(),hover,SaveTool()])
		fig3.line(x=imgprocessarray, y=AvgIOU[index],line_width=1 )
		fig3.xaxis[0].formatter.use_scientific = False
		doc.add_root(fig3)

	for index in range(0,num_of_yolo_layers):
		fig4 = figure(title='Class Yolo Layer '+str(index+1),width=1500, height=400,x_axis_label='Images Processed',y_axis_label='Class of Yolo Layer'+str(index+1),tools=[BoxZoomTool(), ResetTool(),hover,SaveTool()])
		fig4.line(x=imgprocessarray, y=Class[index],line_width=1 )
		fig4.xaxis[0].formatter.use_scientific = False
		doc.add_root(fig4)

	for index in range(0,num_of_yolo_layers):
		fig5 = figure(title='Objectness Yolo Layer '+str(index+1),width=1500, height=400,x_axis_label='Images Processed',y_axis_label='Objectness of Yolo Layer'+str(index+1),tools=[BoxZoomTool(), ResetTool(),hover,SaveTool()])
		fig5.line(x=imgprocessarray, y=Obj[index],line_width=1 )
		fig5.xaxis[0].formatter.use_scientific = False
		doc.add_root(fig5)

	for index in range(0,num_of_yolo_layers):
		fig6 = figure(title='No Objectness Yolo Layer '+str(index+1),width=1500, height=400,x_axis_label='Images Processed',y_axis_label='No Objectness of Yolo Layer'+str(index+1),tools=[BoxZoomTool(), ResetTool(),hover,SaveTool()])
		fig6.line(x=imgprocessarray, y=NoObj[index],line_width=1 )
		fig6.xaxis[0].formatter.use_scientific = False
		doc.add_root(fig6)

	for index in range(0,num_of_yolo_layers):
		fig7 = figure(title='.5 IOU Recall Yolo Layer '+str(index+1),width=1500, height=400,x_axis_label='Images Processed',y_axis_label='.5 IOU Recall of Yolo Layer'+str(index+1),tools=[BoxZoomTool(), ResetTool(),hover,SaveTool()])
		fig7.line(x=imgprocessarray, y=FIFTYR[index],line_width=1 )
		fig7.xaxis[0].formatter.use_scientific = False
		doc.add_root(fig7)

	for index in range(0,num_of_yolo_layers):
		fig8 = figure(title='.75 IOU Recall Yolo Layer '+str(index+1),width=1500, height=400,x_axis_label='Images Processed',y_axis_label='.75 IOU Recall of Yolo Layer'+str(index+1),tools=[BoxZoomTool(), ResetTool(),hover,SaveTool()])
		fig8.line(x=imgprocessarray, y=SEVENFIVER[index],line_width=1 )
		fig8.xaxis[0].formatter.use_scientific = False
		doc.add_root(fig8)

		doc.title = "Darknet Visualization!"

mchost = sys.argv[1]
server = Server(modify_doc, io_loop=IOLoop(), allow_websocket_origin=[mchost+":5006"])
server.start()
server.io_loop.start()