<template>
  <div class="common-layout">
            <form @submit.prevent="submit">
              <p class="text">文章  <el-input v-model="passage" :rows="10" type="textarea" placeholder="Please input"/></p>
            </form>
            <el-button type="success" plain @click="submit">提交</el-button>
            <div>
              <p class="text">title: {{title}}</p>
              <p class="text">abstract: {{abstract}}</p>
            </div>
  </div>

</template>

<script>
import axios from 'axios'
export default {
  name: 'HelloWorld',
  data() {
    return {
      passage: "",
      abstract: "",
      title:""
    }},
    methods: {
      submit()
      {
        //console.log(this.passage);
        //let that = this;
        console.log(JSON.parse(JSON.stringify(this.passage)));
        axios({
          method: 'post',
          url: 'http://127.0.0.1:5000',
          data: {
            passage: this.passage
          },
          headers: {
            'Content-Type':'application/json'
          }
        }).then((res) => {
          console.log('数据：', res.data.data)
          this.abstract = res.data.data['abstract'];
          this.title = res.data.data['title'];
        }).catch(err => {
          console.log('请求失败：'+err.status+','+err.statusText);
        });
      }
    }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.common-layout{
  margin-top: -50px;
}

.text{
  font-family: JYDJW;
  font-size: x-large;
}
.Aside{
  background-color:  #337ecc;
}
.Main{
  background-color:  #79bbff;
}
.Header{
  background-color:  #c6e2ff;
}
</style>
