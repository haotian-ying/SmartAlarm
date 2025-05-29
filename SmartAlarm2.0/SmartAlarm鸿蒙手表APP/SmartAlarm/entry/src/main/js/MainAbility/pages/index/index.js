import router from '@ohos.router'
export default {
    data: {
        title: ""
    },
    onInit() {

    },
    btnClick(){
        router.replaceUrl({
            uri: 'pages/time/time'
        })
    }
}
