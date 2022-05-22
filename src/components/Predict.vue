<template>
  <el-alert v-show="err==1" title="error alert" type="error" @close="close"/>
  <p class="fromRight" style="font-family: JYDJW; font-size: 30px">选择您所想要分析的数据</p>
  <div class="selectItem">
  <el-form-item label="金融投资品一" style="margin:10px" >
    <el-select v-model="form.item1" placeholder="please select">
      <el-option label="黄金" value="黄金" />
      <el-option label="比特币" value="比特币" />
      <el-option label="石油" value="石油" />
      <el-option label="上证50" value="上证50" />
      <el-option label="沪深300" value="沪深300" />
      <el-option label="中证500" value="中证500" />
    </el-select>
  </el-form-item>
  <el-form-item label="金融投资品二" style="margin:10px">
    <el-select v-model="form.item2" placeholder="please select">
      <el-option label="黄金" value="黄金" />
      <el-option label="比特币" value="比特币" />
      <el-option label="石油" value="石油" />
      <el-option label="上证50" value="上证50" />
      <el-option label="沪深300" value="沪深300" />
      <el-option label="中证500" value="中证500" />
    </el-select>
  </el-form-item>
  </div>
  <div>
    <el-row style="display: flex;align-content: center;justify-content: center" class="fromLeft">
      <p>请输入您的本金<el-input v-model="USD" placeholder="Please input" clearable style="width: 300px;margin: 10px"/></p>
      <el-form-item label="单位" style="margin:20px; width: 200px">
        <el-select v-model="MoneyForm" placeholder="please select">
          <el-option label="人民币" value="黄金" />
          <el-option label="美元" value="比特币" />
        </el-select>
      </el-form-item>

    </el-row>

    <p class="fromRight">请输入您持有{{form.item1}}的量<el-input v-model="item1money" placeholder="Please input" clearable style="width: 500px;margin: 10px"/></p>
    <p class="fromLeft">请输入您持有{{form.item2}}的量<el-input v-model="item2money" placeholder="Please input" clearable style="width: 500px;margin: 10px"/></p>
  </div>
  <div class="fromRight">
    <el-radio v-model="choose" label="1">稳健</el-radio>
    <el-radio v-model="choose" label="2">经典</el-radio>
    <el-radio v-model="choose" label="3">激进</el-radio>
  </div>
  <el-button class="fromLeft" type="primary" plain style="margin-top: 50px" @click="submit" :disabled="isabled">开始预测</el-button>
<!--  <p>{{whatToDo}}</p>-->
  <el-drawer
      v-model="drawer"
      title="投资建议"
      :direction="direction"
  >
    <span><p style="font-family: XYSJW">{{passage1}}</p></span>
  </el-drawer>
<!--  <button @click="drawer = true">hhh</button>-->
</template>

<script>
import { reactive } from 'vue'
import axios from 'axios'
/* eslint-disable */
export default {
  name: "Predict",
  data() {
    return {
      form : reactive({
        item1:"商品",
        item2:"商品"
      }),
      err:0,
      USD:"",
      item1money:"",
      item2money:"",
      MoneyForm:"",
      choose:"2",
      whatToDo:"",
      buyItem1:"",
      buyItem2:"",
      showAns:0,
      isabled:false,
      drawer:false,
      direction:"btt",
      passage1:"",
      passage2:"",
      passage3:"",
    }},
  methods:{
    submit(){
      if(this.form.item1 == this.form.item2)
      {
        this.err = 1;
      }
      else
      {
        this.isabled = true;
        this.showAns = 0;
        this.err = 0;
        alert("正在预测")
        axios({
          method: 'post',
          url: 'http://127.0.0.1:5000/predict',
          data: {
            item1: this.form.item1,
            item2: this.form.item2,
            USD:this.USD,
            item1money:this.item1money,
            item2money:this.item2money,
            MoneyForm:this.MoneyForm,
            choose:this.choose
          },
          headers: {
            'Content-Type': 'application/json'
          }
        }).then((res) => {
          console.log('数据：', res.data.data)
          this.whatToDo = res.data.data['whatToDo'];
          this.buyItem1 = res.data.data['buyItem1'];
          this.buyItem2 = res.data.data['buyItem2'];
          this.showAns = 1;
          this.isabled = false;
          this.drawer = true;
          if(this.whatToDo == "1"){
            this.passage1 = "啥也别动";
          }
          else if(this.whatToDo == "2")
          {
            this.passage1 = "购买产品" + this.form.item1 + this.buyItem1 * this.USD;
          }
          if(this.whatToDo == "3"){
            this.passage1 = "购买产品" + this.form.item2 + this.buyItem2 * this.USD;
          }
          else if(this.whatToDo == "4")
          {
            this.passage1 = "多少都买点" + this.form.item1 + this.buyItem1 * this.USD + this.form.item2+this.buyItem2 * this.USD ;
          }
          else if(this.whatToDo == "5"){
            this.passage1 = "卖出产品" + this.form.item1 + -this.buyItem1 * this.USD;
          }
          else if(this.whatToDo == "6")
          {
            this.passage1 = "卖出产品" + this.form.item2+ -this.buyItem2 * this.USD;
          }
          if(this.whatToDo == "7"){
            this.passage1 = "多少卖点"+ this.form.item1 + -this.buyItem1 * this.USD + this.form.item2 + -this.buyItem2 * this.USD;
          }
          else if(this.whatToDo == "8")
          {
            this.passage1 = "买" + this.form.item1 + this.buyItem1 * this.USD +"卖" + this.form.item2 + -this.buyItem2 * this.USD ;
          }
          else if(this.whatToDo == "9")
          {
            this.passage1 = "卖" + this.form.item1 + -this.buyItem1 * this.USD +"买"+ this.form.item2 + this.buyItem2 * this.USD ;
          }
        }).catch(err => {
          console.log('请求失败：' + err.status + ',' + err.statusText);
        });
      }
    },
    close()
    {
      this.err = 0;
    }
  }
}
</script>

<style scoped>
.selectItem{
  display: flex;
  align-content: center;
  justify-content: center;
}
.fromLeft{
  animation: bounceInLeft; /* referring directly to the animation's @keyframe declaration */
  animation-duration: 2s; /* don't forget to set a duration! */
}
.fromRight{
  animation: bounceInRight; /* referring directly to the animation's @keyframe declaration */
  animation-duration: 2s; /* don't forget to set a duration! */
}
</style>