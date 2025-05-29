import router from '@ohos.router';
export default {
    data: {
        time: "",
        dayarray:["上午","下午"],
        hourarray:["01", "02", "03", "04", "05", "06", "07", "08", "09","10","11","12"],
        minutearray:["00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
                    "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                    "20", "21", "22", "23", "24", "25", "26", "27", "28","29",
                    "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
                    "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
                    "50", "51", "52", "53", "54", "55", "56", "57", "58", "59"],
        dataWrapper: {
            mode: "",
            time: "00:00",
            repeat: "不重复",
            switchStatus:false,
            alarmItemIndex: -1
        },
        // 选中值的index
        dayselect : 0,
        hourselect : 11,
        minuteselect : 0,
        targetHour: "00",
        targetMinute: "00",
        targeRepeat: '不重复',
        originData:[]
    },
    onInit() {
        // 转换为 12 小时制
        console.log(this.dataWrapper.mode);
        if(this.dataWrapper.mode == "editAlarm" || this.dataWrapper.mode == "addAlarm") {
            var hours = Number(this.dataWrapper.time.substr(0, 2));
            console.log(hours);
            var minutes = Number(this.dataWrapper.time.substr(3, 2));
            console.log(minutes);
            var day = parseInt(hours / 12);
            hours = hours % 12;
            hours = hours ? hours : 12;
            minutes = minutes;
            this.dayselect = day;
            this.hourselect = hours - 1;
            this.minuteselect = minutes;
        }
    },
    dayChange(e){
        this.dayselect = e.newSelected;
        // console.log(this.dayselect);
    },
    hourChange(e){
        this.hourselect = e.newSelected;
        // console.log(this.hourselect);
    },
    minuteChange(e){
        this.minuteselect = e.newSelected;
        // console.log(this.minuteselect);
    },
    confirmTime(){
        var day = this.dayselect;
        console.log(day);
        var hour = this.hourselect + 1;
        var minute = this.minuteselect;
        //console.log(this.minuteselect);
        //console.log(minute);
        // console.log(this.minuteselect);
        if(day == 0 && hour == 12){
            hour = 0;
        }
        if(day == 1){
            hour += 12;
        }

        this.targetHour = hour;
        this.targetMinute = minute;

        if(hour < 10){
            this.targetHour = '0' + hour;
        }
        if(minute < 10){
            this.targetMinute = '0' + minute;
        }
        // console.log(this.targetHour + " : " + this.targetMinute );
        // console.log(this.hourselect);
        // console.log(this.minuteselect);

        //console.log(minute);
        this.dataWrapper.time = this.targetHour + ':' + this.targetMinute;
        // 返回并增加闹钟
        console.log(JSON.stringify(this.dataWrapper));
        switch(this.dataWrapper.mode){
            case "editAlarm":
                router.replaceUrl({
                    uri: 'pages/time/time',
                    params: {
                        dataWrapper: this.dataWrapper,
                        originData: this.originData
                    }
                });
                break;
            case "addAlarm":
                router.replaceUrl({
                    uri: 'pages/editRepeat/editRepeat',
                    params: {
                        dataWrapper: this.dataWrapper,
                        originData: this.originData
                    }
                });
            default:
                break;
        }
    },
}