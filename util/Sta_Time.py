import time

def sta_show_now_time_msg(msg=''):
    t = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print(f'{t} {msg}')
    t_start = time.time()
    return t_start

def sta_get_start_time():
    t_start = time.time()
    return t_start

def sta_show_used_time_msg(t_start, **kwargs):
    msg = ''
    if('msg' in kwargs.keys()):
        msg = kwargs['msg']

    t_end = time.time()
    t_show = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_end))
    t_diff = t_end - t_start  #second
    hour = 0
    minute = 0
    second = 0

    leftTime = t_diff
    hour = leftTime//3600
    if(hour > 0):
        leftTime = leftTime%3600
    else:
        leftTime = leftTime
    
    minute = leftTime//60
    if(minute > 0):
        leftTime = leftTime%60
    else:
        leftTime = leftTime

    hour = int(hour)
    minute = int(minute)
    second = round(leftTime,4)

    print(f'{t_show} {msg} used time: {hour}h-{minute}m-{second}s')