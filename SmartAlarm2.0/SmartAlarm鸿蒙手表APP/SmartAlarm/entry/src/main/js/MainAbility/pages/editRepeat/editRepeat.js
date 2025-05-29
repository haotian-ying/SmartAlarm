import router from '@system.router';
export default {
    data: {
        Weeks:[
            {weekOn:false,week:'日'},
            {weekOn:false,week:'一'},
            {weekOn:false,week:'二'},
            {weekOn:false,week:'三'},
            {weekOn:false,week:'四'},
            {weekOn:false,week:'五'},
            {weekOn:false,week:'六'}
        ],
        dataWrapper: {
            mode: "",
            time: "00:00",
            repeat: "不重复",
            switchStatus:false,
            alarmItemIndex: -1
        },
        targeRepeat: '不重复',
        originData:[]
    },
    onInit() {
        if (this.dataWrapper.mode === "editAlarm") {
            for (var i = 0; i < this.Weeks.length; i++) {
                if (this.dataWrapper.repeat.indexOf(this.Weeks[i].week) !== -1) {
                    this.Weeks[i].weekOn = true;
                }
            }
        }
    },
    changeWeekSelected(index){
        // 重复点击可取消
        this.Weeks[index].weekOn = this.Weeks[index].weekOn? false : true;
    },
    getRepeat(){
        let repeat = '';
        for (var index = 0; index < this.Weeks.length; index++) {
            if(this.Weeks[index].weekOn){
                repeat = repeat + this.Weeks[index].week + ' ';
            }
        }
        if(repeat == ''){
            repeat = '不重复';
        }
        this.dataWrapper.repeat = repeat;
    },
    confirm(){
        this.getRepeat();
        console.log(JSON.stringify(this.dataWrapper));
        router.replace({
            uri: 'pages/time/time',
            params: {
                dataWrapper: this.dataWrapper,
                originData: this.originData
            }
        });
    }
}
