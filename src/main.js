import  { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import "@/assets/text/text.css"
import {router} from './router/index'
import 'animate.css';
import VueVideoPlayer from 'vue-video-player'
import 'video.js/dist/video-js.css'
// require('vue-video-player/src/theme/myVideo.css')


const app = createApp(App)
app.use(VueVideoPlayer)

app.use(ElementPlus)
app.use(router)
app.mount('#app')