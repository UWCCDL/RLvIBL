console.log('model_bar.js')
function convertModelData0(mData, id) {
    return {
        labels: ['M1', 'M2'], 
        datasets:[{
            data: [mData[id]['m1'].RewardBlock.RewardTrial, mData[id]['m1'].RewardBlock.LossTrial],
            label:["M1: Reward Block"],   // legend name 
            fill:true,    // fill areas below line
            //borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(222, 45, 38, 0.5)',
            // order: 1,
            // barPercentage: 0.8,
            // barThickness: 10,
            // maxBarThickness: 10,
            // minBarLength: 10,
        }, 
        {   data: [mData[id]['m1'].LossBlock.RewardTrial, mData[id]['m1'].LossBlock.LossTrial],
            label:["M1: Loss Block"],   // legend name 
            fill:false,    // fill areas below line
            borderColor: '#fff',
            backgroundColor: 'rgba(222, 45, 38, 0.5)',
            // order: 1,
            // barPercentage: 0.8,
            // barThickness: 10,
            // maxBarThickness: 10,
            // minBarLength: 10
        },
        {   data: [mData[id]['m2'].RewardBlock.LossTrial, mData[id]['m2'].LossBlock.LossTrial],
            label:["Reward Trial", "Loss Trial"],   // legend name 
            fill:false,    // fill areas below line
            borderColor: '#fff',
            backgroundColor: 'rgba(49, 130, 189, 0.5)',
            order: 1,
            // barPercentage: 0.5,
            // barThickness: 10,
            // maxBarThickness: 10,
            // minBarLength: 10
        },
        {   data: [mData[id]['m2'].RewardBlock.LossTrial, mData[id]['m2'].LossBlock.LossTrial],
            label:["Reward Trial", "Loss Trial"],   // legend name 
            fill:false,    // fill areas below line
            //borderColor: '#fff',
            backgroundColor: 'rgba(49, 130, 189, 0.5)',
            order: 1,
            // barPercentage: 0.5,
            // barThickness: 10,
            // maxBarThickness: 10,
            // minBarLength: 10
        }]
    }
}


function convertModelData(mData, id) {
    return {
        labels: ['Reward Trial - Reward Block', 'Loss Trial - Reward Block', 'Reward Trial - Loss Block', 'Loss Trial - Loss Block'], 
        datasets:[{
            data: [mData[id]['m1'].RewardBlock.RewardTrial, mData[id]['m1'].RewardBlock.LossTrial, mData[id]['m1'].LossBlock.RewardTrial, mData[id]['m1'].LossBlock.LossTrial],
            label:"Model 1",   // legend name 
            fill:false,    // fill areas below line
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(222, 45, 38, 0.5)',
            order: 1,
            barPercentage: 0.5
        }, 
        {   data: [mData[id]['m2'].RewardBlock.RewardTrial, mData[id]['m2'].RewardBlock.LossTrial, mData[id]['m2'].LossBlock.RewardTrial, mData[id]['m2'].LossBlock.LossTrial],
            label:"Model 2",   // legend name 
            fill:false,    // fill areas below line
            borderColor: '#fff',
            backgroundColor: 'rgba(49, 130, 189, 0.5)',
            order: 1,
            barPercentage: 0.5
        }]
    }
}

function plotModelData(modelDat, id) {
    //console.log(data)
    const dummy_dataset1 = convertModelData(modelDat, id);
    // const dummy_dataset2 = convertModelData(modelDat, id);
    const config = {
        type:"bar",
        data: dummy_dataset1,
        options:{
          responsive:true,
          plugins: {
            title: {
                display: true,
                text: 'Model 1 Fit Probability of Switching - Subject (' + id + ')'
                },
            subtitle:{
                display: true,
                text: 'Best Fit Model: ' + modelDat[id]['best_model']
            }
            },
            scales: {
                y:{
                    beginAtZero: true,
                    max:1
                }
            },
        },
      };
    
    const ctx = 'model_pswitch'; //objChart.getContext(ctx);
    const chart = new Chart(ctx, config);
    return chart
}

function plotUpdateModelData(chart, modelDat, id) {
    console.log(modelDat[id]);
    const dummy_dataset = convertModelData(modelDat, id);

    chart.options.plugins.title.text = 'Model Fit Probability of Switching - Subject (' + id + ')';
    chart.options.plugins.subtitle.text = 'Best Fit Model: ' + modelDat[id]['best_model'];
    chart.data = dummy_dataset;
    chart.update()
}