import router from '@ohos.router'
export default {
    data: {
        main_if:true,
        add_if:true,
        tips_show:true,
        weeks: ["一", "二", "三", "四", "五", "六", "日"],
        alarm:[
            {time:'07:20',repeat:'一 二 三 四 五',switchStatus:false},
            {time:'08:20',repeat:'一 二 三 四 五',switchStatus:false},
            {time:'13:00',repeat:'日 一 二 三 四 五 六',switchStatus:false},
            {time:'18:00',repeat:'不重复',switchStatus:false}
        ],
        dataWrapper: {
            mode: "",
            time: "00:00",
            repeat: "不重复",
            switchStatus:false,
            alarmItemIndex: -1
        },
        sleeppercent : 82.5,
        sleephour :6,
        sleepminute:36,
        deeppercent:23.4,
        deephour:1,
        deepminute:33,
        interValId: "",
        originData:[]
    },
    onInit() {
        // 判断是那种方式回来的
        switch (this.dataWrapper.mode) {
            // addAlarm editAlarm 均从dataWrapper获得修改
            case 'addAlarm':
            // 把新增的闹钟加进去
                //console.info(this.dataWrapper.mode);
                //console.info(this.alarm[0]);
                this.alarm = this.originData;
                this.alarm.push({time:this.dataWrapper.time,repeat:this.dataWrapper.repeat,switchStatus:this.dataWrapper.switchStatus})
                break;
            case 'editAlarm':
            // 同步修改的值
                this.alarm = this.originData;
                this.alarm[this.dataWrapper.alarmItemIndex].time = this.dataWrapper.time;
                this.alarm[this.dataWrapper.alarmItemIndex].repeat = this.dataWrapper.repeat;
                this.alarm[this.dataWrapper.alarmItemIndex].switchStatus = this.dataWrapper.switchStatus;
                break;
            case 'deleteAlarm':
                this.alarm = this.originData;//同步数据
            // 删除对应位置的闹钟
                this.alarm.splice(this.dataWrapper.alarmItemIndex,1);
                break;
            case 'back':
                this.alarm = this.originData;//同步数据
                break;
            default:
                break;
        }
    },
    addAlarm(){
        // 通过不同的标志判断是什么操作
        this.dataWrapper.mode = 'addAlarm';
        var date = new Date();
        var strHour = date.getHours();
        var strMin = date.getMinutes();
        if(strHour < 10)
            strHour = "0" + strHour;
        if(strMin < 10)
            strMin = "0" + strMin;
        console.log(strMin);
        // 传递当前时间参数
        this.originData = this.alarm;
        this.dataWrapper.time = strHour + ":" + strMin;
        console.log(JSON.stringify(this.dataWrapper));
        router.replaceUrl({
            uri: 'pages/editAlarm/editAlarm',
            params: {
                dataWrapper: this.dataWrapper,
                originData: this.originData
            }
        })
        clearInterval(this.interValId);
    },
    switchToEditAlarm(index) {
        console.log("edit");
        this.dataWrapper.mode = 'editAlarm';
        this.dataWrapper.time = this.alarm[index].time;
        this.dataWrapper.repeat = this.alarm[index].repeat;
        this.dataWrapper.switchStatus = this.alarm[index].switchStatus;
        this.dataWrapper.alarmItemIndex = index;
        // console.log(JSON.stringify(this.dataWrapper));
        this.originData = this.alarm;//同步数据
        router.replaceUrl({
            uri: 'pages/editNavigation/editNavigation',
            params: {
                dataWrapper: this.dataWrapper,
                originData: this.originData
            }
        });
        clearInterval(this.interValId);
    },
    switchChange(index,e){
        console.log("switch");
        this.alarm[index].switchStatus = e.checked;
        this.dataWrapper.mode = 'back';
        this.originData = this.alarm; //同步数据
        router.replaceUrl({
            uri: 'pages/smartConfirm/smartConfirm',
            params: {
                dataWrapper: this.dataWrapper,
                originData: this.originData
            }
        });
    },
    tips(){
        this.tips_show = false;
    }
}
