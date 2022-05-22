// import HelloWorld from "@/components/HelloWorld";
/* eslint-disable */
import {createRouter, createWebHashHistory} from "vue-router";
import home from "@/components/home";
import Predict from "@/components/Predict";
import MyIntro from "@/components/MyIntro";
import APIntro from "@/components/APIntro";

const routes = [
    {
        path: '/home',
        name: 'home',
        component: home
    },
    {
        path: '/intro',
        name:'intro',
        component: MyIntro
    },
    {
        path: '/predict',
        name:'predict',
        component: Predict
    },
    {
        path: '/APIntro',
        name:'APIntro',
        component: APIntro
    },
    {
        path: '/',
        component: home
    }
]

export const router = createRouter({
    history: createWebHashHistory(),
    routes: routes
})