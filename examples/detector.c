#include "darknet.h"
#include "libgen.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};


void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *logdir, char *resume_training)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
   
    struct stat st = {0};

    if (stat(logdir, &st) == -1) {
        mkdir(logdir, 0700);
    }
    srand(time(0));
    network *net = nets[0];
    int imgs = net->batch * net->subdivisions * ngpus;
    char logdircopy[50];
    char lossdump[50],detectiondump[50],lossdumptemp[50],detectiondumptemp[50];
    strcpy(logdircopy,logdir);
    strcat(logdircopy, "/");
    strcpy(lossdump,logdircopy);
    strcpy(lossdumptemp,lossdump);
    strcpy(detectiondump,logdircopy);
    strcpy(detectiondumptemp,detectiondump);
    strcat(lossdump, "lossdump.csv");
    strcat(detectiondump, "detectiondump.csv");
    strcat(lossdumptemp, "lossdumptemp.csv");
    strcat(detectiondumptemp, "detectiondumptemp.csv");
    FILE *lossfile,*detfile;
    FILE *lossfiletemp,*detfiletemp;
    if (strcmp(resume_training,"yes") == 0){
        lossfile = fopen(lossdump, "r");  
        detfile = fopen(detectiondump, "r");
        lossfiletemp = fopen(lossdumptemp, "w");  
        detfiletemp = fopen(detectiondumptemp, "w");
        int line_num = 1;
        int find_result = 0;
        char temp[512];
        char stringtobecomp[50];
        strcpy(stringtobecomp,"Images_Processed:");
        int imgnumfind = net->batch * net->subdivisions * ngpus * get_current_batch(net);
        char charimgnumfind[20];
        sprintf(charimgnumfind,"%d",imgnumfind);
        strcat(stringtobecomp,charimgnumfind);
        strcat(stringtobecomp,"\n");
        // printf("%s\n", stringtobecomp);        
        while(fgets(temp, 512, lossfile) != NULL) {
            fprintf(lossfiletemp,"%s", temp);
            fflush(lossfiletemp);
            if((strstr(temp, stringtobecomp)) != NULL) {
                find_result++;
                break;
            }
        line_num++;
        }
        fclose(lossfiletemp);
        if(find_result == 0) {
            printf("Corrupted Log File :(, give resume_training as no and execute the command again");
            exit(1);
        }
        
        line_num = 1;
        find_result = 0;
 
        while(fgets(temp, 512, detfile) != NULL) {
            fprintf(detfiletemp,"%s", temp);
            fflush(detfiletemp);
            if((strstr(temp, stringtobecomp)) != NULL) {
                find_result++;
                break;
            }
        line_num++;
        }
        fclose(detfiletemp);
        if(find_result == 0) {
            printf("Corrupted Log File :(, give resume_training as no and execute the command again");
            exit(1);
        }
        remove(lossdump);
        rename(lossdumptemp,lossdump);
        remove(lossdumptemp);
        remove(detectiondump);
        rename(detectiondumptemp,detectiondump);
        remove(detectiondumptemp);
    } 
    else if (strcmp(resume_training,"no") == 0){
        lossfile = fopen(lossdump, "w");  
        detfile = fopen(detectiondump, "w");
	    fclose(lossfile);
	    fclose(detfile);
    } 
    printf("Dumping Loss related Log to %s\n", lossdump);
    printf("Dumping Detection related Log to %s\n", detectiondump);

    
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    layer l = net->layers[net->n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    //args.type = INSTANCE_DATA;
    args.threads = 64;

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;



    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
         */
        /*
           int zz;
           for(zz = 0; zz < train.X.cols; ++zz){
           image im = float_to_image(net->w, net->h, 3, train.X.vals[zz]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[zz] + k*5, 1);
           printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
           draw_bbox(im, b, 1, 1,0,0);
           }
           show_image(im, "truth11");
           cvWaitKey(0);
           save_image(im, "truth11");
           }
         */

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

        time=what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train, logdir);
        } else {
            loss = train_networks(nets, ngpus, train, 4,logdir);
        }
#else
        loss = train_network(net, train,logdir);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        FILE *lossfile = fopen(lossdump, "a");
        FILE *detfile = fopen(detectiondump, "a");
        fprintf(lossfile, "Iteration_Number:%ld,Loss:%f,Learning_Rate:%f,Images_Processed:%d\n", get_current_batch(net), loss, get_current_rate(net), i*imgs);
        fflush(lossfile);
        fprintf(detfile, "Iteration_Number:%ld,Loss:%f,Learning_Rate:%f,Images_Processed:%d\n", get_current_batch(net), loss, get_current_rate(net), i*imgs);
        fflush(detfile);


        fclose(lossfile);
        fclose(detfile);

        printf("Iteration_Number:%ld,Loss:%f,Learning_Rate:%f,Images_Processed:%d\n", get_current_batch(net), loss, get_current_rate(net), i*imgs);
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}


static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if(c) p = c;
    return atoi(p+1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for(i = 0; i < num_boxes; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            int class = j;
            if (dets[i].prob[class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, dets[i].prob[class],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector_flip(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 2);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    image input = make_image(net->w, net->h, net->c*2);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data, 1);
            flip_image(val_resized[t]);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data + net->w*net->h*net->c, 1);

            network_predict(net, input.data);
            int w = val[t].w;
            int h = val[t].h;
            int num = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &num);
            if (nms) do_nms_sort(dets, num, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, num, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, num, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, num, classes, w, h);
            }
            free_detections(dets, num);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}


void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, nboxes, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            }
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}

void validate_detector_recall(char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths("data/coco_val_5k.list");
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];

    int j, k;

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = .4;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net->w, net->h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes);
        if (nms) do_nms_obj(dets, nboxes, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < nboxes; ++k){
            if(dets[k].objectness > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < l.w*l.h*l.n; ++k){
                float iou = box_iou(dets[k].bbox, t);
                if(dets[k].objectness > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}


void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];


        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            cvNamedWindow("predictions", CV_WINDOW_NORMAL); 
            if(fullscreen){
                cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            }
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}

/*
void censor_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL); 
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    float nms = .45;

    while(1){
        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        float *X = in_s.data;
        network_predict(net, X);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 0, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int left  = b.x-b.w/2.;
                int top   = b.y-b.h/2.;
                censor_image(in, left, top, b.w, b.h);
            }
        }
        show_image(in, base);
        cvWaitKey(10);
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream(cap);
            free_image(in);
        }
    }
    #endif
}

void extract_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL); 
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    int count = 0;
    float nms = .45;

    while(1){
        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        show_image(in, base);

        int nboxes = 0;
        float *X = in_s.data;
        network_predict(net, X);
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 1, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int size = b.w*in.w > b.h*in.h ? b.w*in.w : b.h*in.h;
                int dx  = b.x*in.w-size/2.;
                int dy  = b.y*in.h-size/2.;
                image bim = crop_image(in, dx, dy, size, size);
                char buff[2048];
                sprintf(buff, "results/extract/%07d", count);
                ++count;
                save_image(bim, buff);
                free_image(bim);
            }
        }
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream(cap);
            free_image(in);
        }
    }
    #endif
}
*/

/*
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets)
{
    network_predict_image(net, im);
    layer l = net->layers[net->n-1];
    int nboxes = num_boxes(net);
    fill_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 0, dets);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
}
*/

void demo_yolo_from_file(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    char   *input;
    float  nms=.3;

    double  start_time;
    double  del_time;
    float  avg_fps=0.0;

    char image_name[256];
    list *plist  = get_paths(filename);
    char **paths = (char **)list_to_array(plist);
    int m        = plist->size;

    int i=0;
    while (i < m) {
        strcpy(image_name,"");

        input    = paths[i];
        image im = load_image_color(input,0,0);

        image sized = letterbox_image(im, net->w, net->h);
        layer l     = net->layers[net->n-1];

        // printf("im.w: %d im.h: %d im.c: %d net->w: %d net->h: %d l.w: %d l.h: %d l.n: %d\n", im.w, im.h, im.c, net->w, net->h, l.w, l.h, l.n);

        float *X = sized.data, fps;

        start_time=what_time_is_it_now();
        network_predict(net, X);
        del_time = what_time_is_it_now() - start_time;

        fps = 1./(del_time);
        if(i==0) avg_fps = fps;
        avg_fps = (avg_fps+fps)/2.0;

        // printf("%s: Predicted in %f seconds. FPS: %f\n", input, del_time, fps);

        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);

        strcat(image_name, "debug/");
        strcat(image_name, basename(input));
        save_image(im, image_name);

        i++;
        free_image(im);
        free_image(sized);
    }

    printf("Average FPS: %f\n", avg_fps);
    free_network(net);
    free_list(plist);
    free(paths);
}

void draw(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    char   *input;

    float nms        = .3;

    double  start_time;
    double  del_time;
    float  avg_fps=0.0;

    char image_name[256];
    list *plist  = get_paths(filename);
    char **paths = (char **)list_to_array(plist);
    int m        = plist->size;

    int i=0, j;

    while (i < m) {
        strcpy(image_name,"");

        input    = paths[i];
        image im = load_image_color(input,0,0);

        image sized = letterbox_image(im, net->w, net->h);
        layer l     = net->layers[net->n-1];

        printf("im.w: %d im.h: %d im.c: %d net->w: %d net->h: %d l.w: %d l.h: %d l.n: %d\n", im.w, im.h, im.c, net->w, net->h, l.w, l.h, l.n);

        float *X = sized.data, fps;

        start_time=what_time_is_it_now();
        network_predict(net, X);
        del_time = what_time_is_it_now() - start_time;

        fps = 1./(del_time);
        if(i==0) avg_fps = fps;
        avg_fps = (avg_fps+fps)/2.0;

        printf("%s: Predicted in %f seconds. FPS: %f\n", input, del_time, fps);

        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);

        // char labelpath[4096];
        // find_replace(input, "images", "labels", labelpath);
        // find_replace(labelpath, ".jpg", ".txt", labelpath);

        // int num_labels = 0;
        // box_label *truth = read_boxes(labelpath, &num_labels);

        // for (j = 0; j < num_labels; ++j) {
        //     box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
        //     draw_bbox(im, t, im.h * 0.006, 0., 0., 1.);
        //     }

        strcat(image_name, "debug/");
        strcat(image_name, basename(input));
        save_image(im, image_name);

        i++;
        free_image(im);
        free_image(sized);
    }

    printf("Average FPS: %f\n", avg_fps);
    free_network(net);
    free_list(plist);
    free(paths);
}

void write_detections(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);
    
    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    
    char   *input;

    float nms        = .25;

    double  start_time;
    double  del_time;
    float  avg_fps=0.0;

    char image_name[256];
    printf("%s\n", filename);

    list *plist  = get_paths(filename);
    char **paths = (char **)list_to_array(plist);
    int m        = plist->size;
    
    int i=0, j, p;

    // char gtsize[20];
    char ptsize[20];
    // char dumpfoldergt[100]; 
    char dumpfolderpt[100]; 

    // sprintf(gtsize,"%d",net->w);
    sprintf(ptsize,"%d",net->w);

    char cfgcopy[50];
    strcpy(cfgcopy,cfgfile);
    char *token1 = strtok(cfgcopy, "/");
    token1 = strtok(NULL, "/");

    char *token2 = strtok(token1, ".");
    printf("%s\n", token2);

    // strcpy(dumpfoldergt,"predictions/adi_score/");
    // strcat(dumpfoldergt, gtsize);
    // strcat(dumpfoldergt, token2);
    // char *gttemp = strcat(dumpfoldergt,"_gt_boxes");
    // char *gtfilename = strcat(gttemp,".txt");
    strcpy(dumpfolderpt,"predictions/adi_score/");
    strcat(dumpfolderpt, ptsize);
    strcat(dumpfolderpt, token2);
    char *pttemp = strcat(dumpfolderpt,"_pt_boxes");
    char *ptfilename = strcat(pttemp,".txt");
    printf("%s\n", cfgfile);
    // printf("%s\n", gtfilename);
    printf("%s\n", ptfilename);


    // FILE *GT = fopen(gtfilename, "w");
    FILE *PT = fopen(ptfilename, "w");
    int detections_count=0;
    while (i < m) {
        strcpy(image_name,"");

        input    = paths[i];
        // [*]
        // fprintf(GT, "%s,", input);
        fprintf(PT, "%s,", input);
        image im = load_image_color(input,0,0);

        image sized = letterbox_image(im, net->w, net->h);
        layer l     = net->layers[net->n-1];

        // printf("im.w: %d im.h: %d im.c: %d net->w: %d net->h: %d l.w: %d l.h: %d l.n: %d\n", im.w, im.h, im.c, net->w, net->h, l.w, l.h, l.n);

        float *X = sized.data, fps;

        start_time=what_time_is_it_now();
        network_predict(net, X);
        del_time = what_time_is_it_now() - start_time;

        fps = 1./(del_time);
        if(i==0) avg_fps = fps;
        avg_fps = (avg_fps+fps)/2.0;

        // printf("%s: Predicted in %f seconds. FPS: %f\n", input, del_time, fps);

        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, 0.005, 0, 0, 1, &nboxes);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        //draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);

        // [*]
        int num_final_boxes=0;
        // [*]
        for(p = 0; p < nboxes; ++p){
            for(j = 0; j < l.classes; ++j){
                if (dets[p].prob[j] > 0.15){
                    box b = dets[p].bbox;
                    //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

                    int left  = (b.x-b.w/2.)*im.w;
                    int right = (b.x+b.w/2.)*im.w;
                    int top   = (b.y-b.h/2.)*im.h;
                    int bot   = (b.y+b.h/2.)*im.h;

                    if(left < 0) left = 0;
                    if(right > im.w-1) right = im.w-1;
                    if(top < 0) top = 0;
                    if(bot > im.h-1) bot = im.h-1;
                    //if(((right - left) > 32 && (bot - top) > 32) ){
                       num_final_boxes++;
                    //}
                }
            }
        }
        
        fprintf(PT, "%d\n", num_final_boxes);
        for(p = 0; p < nboxes; ++p){
            for(j = 0; j < l.classes; ++j){
                if (dets[p].prob[j] > 0.15){
                    box b = dets[p].bbox;
                    //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

                    int left  = (b.x-b.w/2.)*im.w;
                    int right = (b.x+b.w/2.)*im.w;
                    int top   = (b.y-b.h/2.)*im.h;
                    int bot   = (b.y+b.h/2.)*im.h;

                    if(left < 0) left = 0;
                    if(right > im.w-1) right = im.w-1;
                    if(top < 0) top = 0;
                    if(bot > im.h-1) bot = im.h-1;
                    //if(((right - left) > 32 && (bot - top) > 32) ){
                        detections_count++;
                        fprintf(PT, "%d,%d,%d,%d,%f,%d\n", left, top, right, bot, dets[p].prob[j], j);
                        fflush(PT);
                    //}
                }
            }
        }
        // fprintf(PT, "\n");
        fflush(PT);

        // char labelpath[4096];
        // find_replace(input, "images", "labels", labelpath);
        // find_replace(labelpath, ".png", ".txt", labelpath);
        // int num_labels = 0;
        // box_label *truth = read_boxes(labelpath, &num_labels);
        // int act_num_labels = 0;
        // // printf("%s\n",labelpath );
        // // printf("%d\n",num_labels );
        // // [*]
        // for (j = 0; j < num_labels; ++j) {
        //     box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
        //     // [*]
        //     int left  = (t.x-t.w/2)*im.w;
        //     int right = (t.x+t.w/2)*im.w;
        //     int top   = (t.y-t.h/2)*im.h;
        //     int bot   = (t.y+t.h/2)*im.h;

        //     if(left < 0) left = 0;
        //     if(right > im.w-1) right = im.w-1;
        //     if(top < 0) top = 0;
        //     if(bot > im.h-1) bot = im.h-1;
        //     if(j==num_labels-1){
        //         // if((right - left) > 32 && (bot - top) > 32)
        //             act_num_labels ++;
        //     }
        //     else{
        //         // if((right - left) > 32 && (bot - top) > 32)
        //             act_num_labels ++;
        //     }
        // }
        // fprintf(GT, "%d\n", act_num_labels);
        // for (j = 0; j < num_labels; ++j) {
        //     box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
        //     // [*]
        //     int left  = (t.x-t.w/2)*im.w;
        //     int right = (t.x+t.w/2)*im.w;
        //     int top   = (t.y-t.h/2)*im.h;
        //     int bot   = (t.y+t.h/2)*im.h;

        //     if(left < 0) left = 0;
        //     if(right > im.w-1) right = im.w-1;
        //     if(top < 0) top = 0;
        //     if(bot > im.h-1) bot = im.h-1;
        //     if(j==num_labels-1){
        //       //  if((right - left) > 32 && (bot - top) > 32)
        //             fprintf(GT, "%d,%d,%d,%d,0.0,%d\n", left, top, right, bot, truth[j].id);
        //             fflush(GT);
        //     }
        //     else{
        //       //  if(((right - left) > 32 && (bot - top) > 32))
        //             fprintf(GT, "%d,%d,%d,%d,0.0,%d\n", left, top, right, bot, truth[j].id);
        //             fflush(GT);
        //     }
        // }
        // // fprintf(GT, "\n");
        // fflush(GT);

        i++;
        free_image(im);
        free_image(sized);
    }
    printf("total detections: %d\n", detections_count);
    printf("Average FPS: %f\n", avg_fps);
    // fclose(GT);
    free_network(net);
    free_list(plist);
    free(paths);
}

void write_fddb(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    char   *input;

    float nms        = .3;

    double  start_time;
    double  del_time;
    float  avg_fps=0.0;

    char image_name[256];
    list *plist  = get_paths(filename);
    char **paths = (char **)list_to_array(plist);
    int m        = plist->size;

    int i=0, j, p;
    char ptsize[20];
    char dumpfolderpt[100]; 

    char cfgcopy[50];
    strcpy(cfgcopy,cfgfile);
    char *token1 = strtok(cfgcopy, "/");
    token1 = strtok(NULL, "/");

    char *token2 = strtok(token1, ".");
    printf("%s\n", token2);

    sprintf(ptsize,"%d",net->w);

    strcpy(dumpfolderpt,"predictions/fddb_score/");
    strcat(dumpfolderpt, ptsize);
    strcat(dumpfolderpt, token2);
    char *pttemp = strcat(dumpfolderpt,"_pt_boxes");
    char *ptfilename = strcat(pttemp,".txt");

    printf("%s\n", ptfilename);

    FILE *PT = fopen(ptfilename, "w");

    int detections_count=0;
    while (i < m) {
        strcpy(image_name,"");

        input    = paths[i];
        // [*]
        // char splitinput[30];

        // size_t destination_size = sizeof (splitinput);
        // strncpy(splitinput, &input[38], destination_size);
        // printf("%s\n", splitinput);
        // splitinput[destination_size - 1] = '\0';

        // char *token = strtok(splitinput, ".");
        fprintf(PT, "%s\n", input);
        image im = load_image_color(input,0,0);

        image sized = letterbox_image(im, net->w, net->h);
        layer l     = net->layers[net->n-1];

        float *X = sized.data, fps;

        start_time=what_time_is_it_now();
        network_predict(net, X);
        del_time = what_time_is_it_now() - start_time;

        fps = 1./(del_time);
        if(i==0) avg_fps = fps;
        avg_fps = (avg_fps+fps)/2.0;

        
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, 0.005, 0, 0, 1, &nboxes);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        //draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);

        // [*]
        int num_final_boxes=0;
        // [*]
        for(p = 0; p < nboxes; ++p){
            for(j = 0; j < l.classes; ++j){
                if (dets[p].prob[j] > 0.0){
                    num_final_boxes++;
                }
            }
        }
        fprintf(PT, "%d\n", num_final_boxes);
        for(p = 0; p < nboxes; ++p){
            for(j = 0; j < l.classes; ++j){
                if (dets[p].prob[j] > 0.0){
                    detections_count++;
                    box b = dets[p].bbox;
                   
                    int left  = (b.x-b.w/2.)*im.w;
                    int right = (b.x+b.w/2.)*im.w;
                    int top   = (b.y-b.h/2.)*im.h;
                    int bot   = (b.y+b.h/2.)*im.h;

                    if(left < 0) left = 0;
                    if(right > im.w-1) right = im.w-1;
                    if(top < 0) top = 0;
                    if(bot > im.h-1) bot = im.h-1;

                    fprintf(PT, "%d %d %d %d %f\n", left, top, right-left, bot-top, dets[p].prob[j]);
                    fflush(PT);
                }
            }
        }
        // exit(1);
        i++;
        free_image(im);
        free_image(sized);
    }
    printf("total detections: %d\n", detections_count);
    printf("Average FPS: %f\n", avg_fps);
    // fclose(GT);
    free_network(net);
    free_list(plist);
    free(paths);
}

void write_fddb_adi_score(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh)
{
        list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    char   *input;

    float nms        = .25;

    double  start_time;
    double  del_time;
    float  avg_fps=0.0;

    char image_name[256];
    list *plist  = get_paths(filename);
    char **paths = (char **)list_to_array(plist);
    int m        = plist->size;

    int i=0, j, p;

    // char gtsize[20];
    char ptsize[20];
    // char dumpfoldergt[100]; 
    char dumpfolderpt[100]; 

    // sprintf(gtsize,"%d",net->w);
    sprintf(ptsize,"%d",net->w);

    char cfgcopy[50];
    strcpy(cfgcopy,cfgfile);
    char *token1 = strtok(cfgcopy, "/");
    token1 = strtok(NULL, "/");

    char *token2 = strtok(token1, ".");
    printf("%s\n", token2);

    // strcpy(dumpfoldergt,"predictions/adi_score/");
    // strcat(dumpfoldergt, gtsize);
    // strcat(dumpfoldergt, token2);
    // char *gttemp = strcat(dumpfoldergt,"_adi_gt_boxes");
    // char *gtfilename = strcat(gttemp,".txt");

    strcpy(dumpfolderpt,"predictions/adi_score/");
    strcat(dumpfolderpt, ptsize);
    strcat(dumpfolderpt, token2);
    char *pttemp = strcat(dumpfolderpt,"_adi_pt_boxes");
    char *ptfilename = strcat(pttemp,".txt");
    printf("%s\n", cfgfile);
    // printf("%s\n", gtfilename);
    printf("%s\n", ptfilename);


    // FILE *GT = fopen(gtfilename, "w");
    FILE *PT = fopen(ptfilename, "w");
    int detections_count=0;
    while (i < m) {
        strcpy(image_name,"");

        input    = paths[i];
        // [*]
        // fprintf(GT, "%s,", input);
        fprintf(PT, "%s,", input);
        image im = load_image_color(input,0,0);

        image sized = letterbox_image(im, net->w, net->h);
        layer l     = net->layers[net->n-1];

        // printf("im.w: %d im.h: %d im.c: %d net->w: %d net->h: %d l.w: %d l.h: %d l.n: %d\n", im.w, im.h, im.c, net->w, net->h, l.w, l.h, l.n);

        float *X = sized.data, fps;

        start_time=what_time_is_it_now();
        network_predict(net, X);
        del_time = what_time_is_it_now() - start_time;

        fps = 1./(del_time);
        if(i==0) avg_fps = fps;
        avg_fps = (avg_fps+fps)/2.0;

        // printf("%s: Predicted in %f seconds. FPS: %f\n", input, del_time, fps);

        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, 0.005, 0, 0, 1, &nboxes);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        //draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);

        // [*]
        int num_final_boxes=0;
        // [*]
        for(p = 0; p < nboxes; ++p){
            for(j = 0; j < l.classes; ++j){
                if (dets[p].prob[j] > 0.2){
                    box b = dets[p].bbox;
                    //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

                    int left  = (b.x-b.w/2.)*im.w;
                    int right = (b.x+b.w/2.)*im.w;
                    int top   = (b.y-b.h/2.)*im.h;
                    int bot   = (b.y+b.h/2.)*im.h;

                    if(left < 0) left = 0;
                    if(right > im.w-1) right = im.w-1;
                    if(top < 0) top = 0;
                    if(bot > im.h-1) bot = im.h-1;
                    //if(((right - left) > 32 && (bot - top) > 32) ){
                       num_final_boxes++;
                    //}
                }
            }
        }
        
        fprintf(PT, "%d\n", num_final_boxes);
        for(p = 0; p < nboxes; ++p){
            for(j = 0; j < l.classes; ++j){
                if (dets[p].prob[j] > 0.2){
                    box b = dets[p].bbox;
                    //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

                    int left  = (b.x-b.w/2.)*im.w;
                    int right = (b.x+b.w/2.)*im.w;
                    int top   = (b.y-b.h/2.)*im.h;
                    int bot   = (b.y+b.h/2.)*im.h;

                    if(left < 0) left = 0;
                    if(right > im.w-1) right = im.w-1;
                    if(top < 0) top = 0;
                    if(bot > im.h-1) bot = im.h-1;
                    //if(((right - left) > 32 && (bot - top) > 32) ){
                        detections_count++;
                        fprintf(PT, "%d,%d,%d,%d,%f,%d\n", left, top, right, bot, dets[p].prob[j], j);
                        fflush(PT);
                    //}
                }
            }
        }
        // fprintf(PT, "\n");
        fflush(PT);

    //     char labelpath[4096];
    //     find_replace(input, "images", "labels", labelpath);
    //     find_replace(labelpath, ".jpg", ".txt", labelpath);
    //     int num_labels = 0;
    //     box_label *truth = read_boxes(labelpath, &num_labels);
    //     int act_num_labels = 0;
    //     // [*]
    //     for (j = 0; j < num_labels; ++j) {
    //         box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
    //         // [*]
    //         int left  = (t.x-t.w/2)*im.w;
    //         int right = (t.x+t.w/2)*im.w;
    //         int top   = (t.y-t.h/2)*im.h;
    //         int bot   = (t.y+t.h/2)*im.h;

    //         if(left < 0) left = 0;
    //         if(right > im.w-1) right = im.w-1;
    //         if(top < 0) top = 0;
    //         if(bot > im.h-1) bot = im.h-1;
    //         if(j==num_labels-1){
    //             // if((right - left) > 32 && (bot - top) > 32)
    //                 act_num_labels ++;
    //         }
    //         else{
    //             // if((right - left) > 32 && (bot - top) > 32)
    //                 act_num_labels ++;
    //         }
    //     }
    //     fprintf(GT, "%d\n", act_num_labels);
    //     for (j = 0; j < num_labels; ++j) {
    //         box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
    //         // [*]
    //         int left  = (t.x-t.w/2)*im.w;
    //         int right = (t.x+t.w/2)*im.w;
    //         int top   = (t.y-t.h/2)*im.h;
    //         int bot   = (t.y+t.h/2)*im.h;

    //         if(left < 0) left = 0;
    //         if(right > im.w-1) right = im.w-1;
    //         if(top < 0) top = 0;
    //         if(bot > im.h-1) bot = im.h-1;
    //         if(j==num_labels-1){
    //           //  if((right - left) > 32 && (bot - top) > 32)
    //                 fprintf(GT, "%d,%d,%d,%d,0.0,%d\n", left, top, right, bot, truth[j].id);
    //                 fflush(GT);
    //         }
    //         else{
    //           //  if(((right - left) > 32 && (bot - top) > 32))
    //                 fprintf(GT, "%d,%d,%d,%d,0.0,%d\n", left, top, right, bot, truth[j].id);
    //                 fflush(GT);
    //         }
    //     }
    //     // fprintf(GT, "\n");
    //     fflush(GT);

        i++;
        free_image(im);
        free_image(sized);
    }
    printf("total detections: %d\n", detections_count);
    printf("Average FPS: %f\n", avg_fps);
    // fclose(GT);
    free_network(net);
    free_list(plist);
    free(paths);
}

typedef struct {
    box b;
    float p;
    int class_id;
    int image_index;
    int truth_flag;
    int unique_truth_index;
} box_prob;

int detections_comparator(const void *pa, const void *pb)
{
    box_prob a = *(box_prob *)pa;
    box_prob b = *(box_prob *)pb;
    float diff = a.p - b.p;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}
void evaluate_mAP(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh_calc_avg_iou)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    network *net     = load_network(cfgfile, weightfile, 0);
    layer l          = net->layers[net->n-1];
    set_batch_network(net, 1);
    srand(2222222);

    char   *input;

    float iou_thresh = .1;
    float nms        = .3;
    float thresh = .005;

    double  start_time;
    double  del_time;
    float  avg_fps=0.0;

    char image_name[256];
    list *plist  = get_paths(filename);
    char **paths = (char **)list_to_array(plist);
    int m        = plist->size;

    int image_idx=0;

    float avg_iou = 0;
    int tp_for_thresh = 0;
    int fp_for_thresh = 0;

    box_prob *detections = calloc(1, sizeof(box_prob));
    int detections_count = 0;
    int unique_truth_count = 0;

    int *truth_classes_count = calloc(l.classes, sizeof(int));
    while (image_idx < m) {
        strcpy(image_name,"");

        input    = paths[image_idx];
        image im = load_image_color(input,0,0);

        image sized = letterbox_image(im, net->w, net->h);

        float *X = sized.data, fps;

        start_time=what_time_is_it_now();
        network_predict(net, X);
        del_time = what_time_is_it_now() - start_time;

        fps = 1./(del_time);
        if(image_idx==0) avg_fps = fps;
        avg_fps = (avg_fps+fps)/2.0;

        printf("%s: Predicted in %f seconds. FPS: %f\n", input, del_time, fps);

        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, 0, 0, 1, &nboxes);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        char labelpath[4096];
        find_replace(input, "images", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);

        int i, j;
        for (j = 0; j < num_labels; ++j) {
            truth_classes_count[truth[j].id]++;
        }

        int checkpoint_detections_count = detections_count;

        for (i = 0; i < nboxes; ++i) {

            int class_id;
            for (class_id = 0; class_id < l.classes; ++class_id) {
                float prob = dets[i].prob[class_id];
                if (prob > 0) {
                    detections_count++;
                    detections = realloc(detections, detections_count * sizeof(box_prob));
                    detections[detections_count - 1].b = dets[i].bbox;
                    detections[detections_count - 1].p = prob;
                    detections[detections_count - 1].image_index = image_idx;
                    detections[detections_count - 1].class_id = class_id;
                    detections[detections_count - 1].truth_flag = 0;
                    detections[detections_count - 1].unique_truth_index = -1;

                    int truth_index = -1;
                    float max_iou = 0;
                    for (j = 0; j < num_labels; ++j)
                    {
                        box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
                        //printf(" IoU = %f, prob = %f, class_id = %d, truth[j].id = %d \n",
                        //    box_iou(dets[i].bbox, t), prob, class_id, truth[j].id);
                        float current_iou = box_iou(dets[i].bbox, t);
                        if (current_iou > iou_thresh && class_id == truth[j].id) {
                            if (current_iou > max_iou) {
                                max_iou = current_iou;
                                truth_index = unique_truth_count + j;
                            }
                        }
                    }

                    // best IoU
                    if (truth_index > -1) {
                        detections[detections_count - 1].truth_flag = 1;
                        detections[detections_count - 1].unique_truth_index = truth_index;
                    }

                    // calc avg IoU, true-positives, false-positives for required Threshold
                    if (prob > thresh_calc_avg_iou) {
                        int z, found = 0;
                        for (z = checkpoint_detections_count; z < detections_count-1; ++z)
                            if (detections[z].unique_truth_index == truth_index) {
                                found = 1; break;
                            }

                        if(truth_index > -1 && found == 0) {
                            avg_iou += max_iou;
                            ++tp_for_thresh;
                        }
                        else
                            fp_for_thresh++;
                    }
                }
            }
        }

        image_idx++;
        unique_truth_count += num_labels;

        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);
    }

    
    if((tp_for_thresh + fp_for_thresh) > 0)
        avg_iou = avg_iou / (tp_for_thresh + fp_for_thresh);


    // SORT(detections)
    qsort(detections, detections_count, sizeof(box_prob), detections_comparator);

    typedef struct {
        double precision;
        double recall;
        int tp, fp, fn;
    } pr_t;

    // for PR-curve
    pr_t **pr = calloc(l.classes, sizeof(pr_t*));
    int i;
    for (i = 0; i < l.classes; ++i) {
        pr[i] = calloc(detections_count, sizeof(pr_t));
    }
    printf("detections_count = %d, unique_truth_count = %d  \n", detections_count, unique_truth_count);


    int *truth_flags = calloc(unique_truth_count, sizeof(int));

    int rank;
    for (rank = 0; rank < detections_count; ++rank) {
        if(rank % 100 == 0)
            printf(" rank = %d of ranks = %d \r", rank, detections_count);

        if (rank > 0) {
            int class_id;
            for (class_id = 0; class_id < l.classes; ++class_id) {
                pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
                pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
            }
        }

        box_prob d = detections[rank];
        // if (detected && isn't detected before)
        if (d.truth_flag == 1) {
            if (truth_flags[d.unique_truth_index] == 0)
            {
                truth_flags[d.unique_truth_index] = 1;
                pr[d.class_id][rank].tp++;    // true-positive
            }
        }
        else {
            pr[d.class_id][rank].fp++;    // false-positive
        }

        for (i = 0; i < l.classes; ++i)
        {
            const int tp = pr[i][rank].tp;
            const int fp = pr[i][rank].fp;
            const int fn = truth_classes_count[i] - tp;    // false-negative = objects - true-positive
            pr[i][rank].fn = fn;

            if ((tp + fp) > 0) pr[i][rank].precision = (double)tp / (double)(tp + fp);
            else pr[i][rank].precision = 0;

            if ((tp + fn) > 0) pr[i][rank].recall = (double)tp / (double)(tp + fn);
            else pr[i][rank].recall = 0;
        }
    }

    free(truth_flags);


    double mean_average_precision = 0;

    for (i = 0; i < l.classes; ++i) {
        double avg_precision = 0;
        int point;
        for (point = 0; point < 11; ++point) {
            double cur_recall = point * 0.1;
            double cur_precision = 0;
            for (rank = 0; rank < detections_count; ++rank)
            {
                if (pr[i][rank].recall >= cur_recall) {    // > or >=
                    if (pr[i][rank].precision > cur_precision) {
                        cur_precision = pr[i][rank].precision;
                    }
                }
            }
            //printf("class_id = %d, point = %d, cur_recall = %.4f, cur_precision = %.4f \n", i, point, cur_recall, cur_precision);

            avg_precision += cur_precision;
        }
        avg_precision = avg_precision / 11;
        printf("class_id = %d, name = %s, \t ap = %2.2f %% \n", i, names[i], avg_precision*100);
        mean_average_precision += avg_precision;
    }

    const float cur_precision = (float)tp_for_thresh / ((float)tp_for_thresh + (float)fp_for_thresh);
    const float cur_recall = (float)tp_for_thresh / ((float)tp_for_thresh + (float)(unique_truth_count - tp_for_thresh));
    const float f1_score = 2.F * cur_precision * cur_recall / (cur_precision + cur_recall);
    printf(" for thresh = %1.2f, precision = %1.2f, recall = %1.2f, F1-score = %1.2f \n",
        thresh_calc_avg_iou, cur_precision, cur_recall, f1_score);

    printf(" for thresh = %0.2f, TP = %d, FP = %d, FN = %d, average IoU = %2.2f %% \n",
        thresh_calc_avg_iou, tp_for_thresh, fp_for_thresh, unique_truth_count - tp_for_thresh, avg_iou * 100);

    mean_average_precision = mean_average_precision / l.classes;
    printf("\n mean average precision (mAP) = %f, or %2.2f %% \n", mean_average_precision, mean_average_precision*100);


    for (i = 0; i < l.classes; ++i) {
        free(pr[i]);
    }

    printf("Average FPS: %f\n", avg_fps);

    free(pr);
    free(detections);
    free(truth_classes_count);

    free_network(net);
    free_list(plist);
    free(paths);
}

void run_detector(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);
    //int class = find_int_arg(argc, argv, "-class", 0);

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = find_char_arg(argc, argv, "-weights", 0);
    char *filename = find_char_arg(argc, argv, "-image_name_file", 0);
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "file_demo")) demo_yolo_from_file(datacfg, cfg, weights, filename, thresh, hier_thresh);
    else if(0==strcmp(argv[2], "evaluate")) evaluate_mAP(datacfg, cfg, weights, filename, thresh);
    else if(0==strcmp(argv[2], "draw")) draw(datacfg, cfg, weights, filename, thresh, hier_thresh);
    else if(0==strcmp(argv[2], "write")) write_detections(datacfg, cfg, weights, filename, thresh, hier_thresh);
    else if(0==strcmp(argv[2], "fddb")) write_fddb(datacfg, cfg, weights, filename, thresh, hier_thresh);
    else if(0==strcmp(argv[2], "fddb-adi")) write_fddb_adi_score(datacfg, cfg, weights, filename, thresh, hier_thresh);
    else if(0==strcmp(argv[2], "train")){
        char *logdir = find_char_arg(argc, argv, "-logdir", "logdir");
        char *resume_training = find_char_arg(argc, argv, "-resume_training", "no");
        printf("%s\n", logdir);
        train_detector(datacfg, cfg, weights, gpus, ngpus, clear,logdir,resume_training);
    }
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "valid2")) validate_detector_flip(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
    //else if(0==strcmp(argv[2], "extract")) extract_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
    //else if(0==strcmp(argv[2], "censor")) censor_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
}
