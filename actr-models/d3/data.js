console.log('data.js');

var LLData = [];
var LLDataSorted = [];

var behData = [];
var mData = [];

var currID = 0;
var subjectChart = null;
var modelCharts = null;

const url = 'https://raw.githubusercontent.com/UWCCDL/RLvIBL/actr/actr-models/model_output/MODELLogLikelihood.csv';
d3.csv(url)
.then(function(data){
    data.forEach(function(d, i) {
        //console.log(d['PSwitch.LL.m1'] + " " + i);
        // console.log(d)
        d['PSwitch.LL.m1'] = +d['PSwitch.LL.m1'];
        d['PSwitch.LL.m2'] = +d['PSwitch.LL.m2'];
        d['LLDiff'] = d['PSwitch.LL.m1']-d['PSwitch.LL.m2'];
        // if(d.LLDiff != 0){
        //     if (d.LLDiff > 0) {
        //         d.color = '#fee0d2'
        //     } else{
        //         d.color = '#deebf7' 
        //     }
        // } else {
        //     d.color = '#636363' 
        // }

        // save data
        LLData.push({'index':i, 
            'm1':d['PSwitch.LL.m1'], 
            'm2':d['PSwitch.LL.m2'], 
            'diff':d['LLDiff'],
            'color':d.color
        });

        behData.push({
            'index':i, 
            'RewardBlock':{
                'RewardTrial':+d['Reward_MostlyReward.subj'],
                'LossTrial':+d['Punishment_MostlyReward.subj']
            },
            'LossBlock':{
                'RewardTrial':+d['Reward_MostlyPunishment.subj'],
                'LossTrial':+d['Punishment_MostlyPunishment.subj']
            }
        });

        mData.push({
            'index':i, 
            'best_model': d['best_model'],
            'm1':{
                'RewardBlock':{
                    'RewardTrial':+d['PSwitch.mean_Reward_MostlyReward.m1'],
                    'LossTrial':+d['PSwitch.mean_Punishment_MostlyReward.m1']
                },
                'LossBlock':{
                    'RewardTrial':+d['PSwitch.mean_Reward_MostlyPunishment.m1'],
                    'LossTrial':+d['PSwitch.mean_Punishment_MostlyPunishment.m1']
                }
            },
            'm2':{
                'RewardBlock':{
                    'RewardTrial':+d['PSwitch.mean_Reward_MostlyReward.m2'],
                    'LossTrial':+d['PSwitch.mean_Punishment_MostlyReward.m2']
                },
                'LossBlock':{
                    'RewardTrial':+d['PSwitch.mean_Reward_MostlyPunishment.m2'],
                    'LossTrial':+d['PSwitch.mean_Punishment_MostlyPunishment.m2']
                }
            }
        })
    });
    //LLData = LLData.slice().sort((a, b) => d3.descending(a.diff, b.diff))
    LLData = LLData.slice().sort((a, b) => d3.descending(Math.abs(a.diff), Math.abs(b.diff)));
    LLData.forEach(function(d, i) {
        LLDataSorted.push({'sortedIndex':i, 'index':d['index'], 'm1':d['m1'], 'm2':d['m2'], 'diff':d['diff']})
    });
    
    // console.log(LLData)
    plotChart(LLDataSorted);
    plotHist1(LLDataSorted);
    plotHist2(LLDataSorted);
    subjectChart = plotBeh(behData, 0);
    modelCharts = plotModelData(mData, 0);
    
});




plotHist0 = function(data) {
    console.log(data[0])
    // plot chart
    const dummy_dataset  = {
      labels: data.map(d=>d.diff),
      datasets:[{
        type:'bar',
        data:data.map(d=>d.index),
        label:"LL Diff",   // legend name 
        fill:false,    // fill areas below line
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.8)',
        //backgroundColor: '#fff',
        //pointBackgroundColor: 'yellow',
        order: 1,
        barPercentage: 1,
        categoryPercentage:1,
        //barThickness: 6,
        //maxBarThickness: 8,
        //minBarLength: 2,
      }]
    }
    const config = {
      type:"bar",
      data: dummy_dataset,
      options:{
        responsive:true,
        scales:{
            x:{
                type:'linear',
                offset: false,
                grid: {
                    offset: false
                },
                title:{
                    display: true,
                    text: 'Maximum LogLikelihood'
                }
            },
            y:{
                title:{
                    display: true,
                    text: 'Distribution of Max LL Distribution'
                }
            }
        }
        //radius:5,
        //hitRadius:30,
        //hoverRadius:20,
        // animation
      }
    }
    const ctx1 = 'mychart1';
    const mychart1 = new Chart(ctx1, config)
  };
