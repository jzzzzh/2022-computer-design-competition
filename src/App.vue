<template>

  <div class="common-layout">
    <el-container>
      <el-header class="header"><MyHeader></MyHeader></el-header>
      <el-main class="main">  <router-view></router-view></el-main>
    </el-container>
    <el-backtop :right="100" :bottom="100" />
  </div>

</template>

<script>
// import HelloWorld from './components/HelloWorld.vue'
import MyHeader from "@/components/MyHeader";
export default {
  name: 'App',
  components: {
    // HelloWorld
    MyHeader
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}

body{
  margin:0;
  padding:0;
  border:0
}

.header{
  margin-top: -60px;
  background-color:  #c6e2ff;
  height: 80px;
}
.dividers{
  margin-top: 0px;
}

</style>
