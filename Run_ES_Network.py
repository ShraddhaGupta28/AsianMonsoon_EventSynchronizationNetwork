# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
import sys
from build.Event_Sync_Null_Model_Cy import event_sync_null_model
from build.Event_Sync2_Null_Model_Cy import event_sync_null_model as event_sync_null_model2
from build.Event_Sync_Udw_Cy import event_sync
from build.Task1_Ud_ES_Construction import network
from build.Task1_Ud_ES_Regional_Sync_Corr import regional_sync


if __name__ == '__main__':
    # event synchronization ==========================
    event_sync_null_model()
    event_sync_null_model2()
    event_sync()
    network()
    regional_sync()
    print('Hello, my friend..........................')
