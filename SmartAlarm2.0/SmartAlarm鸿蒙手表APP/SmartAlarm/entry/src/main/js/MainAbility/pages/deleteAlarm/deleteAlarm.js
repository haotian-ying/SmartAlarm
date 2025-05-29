import router from '@ohos.router';
export default {
    data: {
        dataWrapper: {
            mode: "",
            time: "00:00",
            repeat: "不重复",
            switchStatus:false,
            alarmItemIndex: -1
        },
        originData:[]
    },
    onInit() {
        console.log(this.dataWrapper.mode);
    },
    cancel(){
        this.dataWrapper.mode = "cancelDelete";
        router.replaceUrl({
            uri: 'pages/time/time',
            params: {
                dataWrapper: this.dataWrapper,
                originData: this.originData
            }
        });
    },
    confirm(){
        router.replaceUrl({
            uri: 'pages/time/time',
            params: {
                dataWrapper: this.dataWrapper,
                originData: this.originData
            }
        });
    }
}
