import os
import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)
import math
import nltk
from scipy.io.wavfile import write
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser
import numpy as np
# import speech_utils as tool

from robot_control.unitree_dog_control import HighStateHandler,SportModeTest

robot_state = unitree_go_msg_dds__SportModeState_()
if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
else:
        ChannelFactoryInitialize(0)
        
sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
sub.Init(HighStateHandler, 10)
time.sleep(1)

test = SportModeTest()
test.GetInitState(robot_state)

test.Stand()
