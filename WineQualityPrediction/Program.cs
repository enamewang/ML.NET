using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;

namespace WineQualityPrediction
{
    class Program
    {
        const string DataPath = @".\Data\winequality-data.csv";
        const string TestDataPath = @".\Data\winequality-test-data.csv";

        public class WineData
        {
            [Column(ordinal: "0")]
            public float FixedAcidity;
            [Column(ordinal: "1")]
            public float VolatileAcidity;
            [Column(ordinal: "2")]
            public float CitricACID;
            [Column(ordinal: "3")]
            public float ResidualSugar;
            [Column(ordinal: "4")]
            public float Chlorides;
            [Column(ordinal: "5")]
            public float FreeSulfurDioxide;
            [Column(ordinal: "6")]
            public float TotalSulfurDioxide;
            [Column(ordinal: "7")]
            public float Density;
            [Column(ordinal: "8")]
            public float PH;
            [Column(ordinal: "9")]
            public float Sulphates;
            [Column(ordinal: "10")]
            public float Alcohol;
            [Column(ordinal: "11", name: "Label")]
            public float Quality;
            [Column(ordinal: "12")]
            public float Id;
        }

        public class WinePrediction
        {
            [ColumnName("Score")]
            public float PredictionQuality;
        }

        static PredictionModel<WineData, WinePrediction> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new Microsoft.ML.Data.TextLoader(DataPath).CreateFrom<WineData>(useHeader: true, separator: ',', trimWhitespace: false));
            pipeline.Add(new ColumnDropper() { Column = new[] { "Id" } });
            pipeline.Add(new ColumnConcatenator("Features", "FixedAcidity", "VolatileAcidity", "CitricACID", "ResidualSugar", "Chlorides", "FreeSulfurDioxide", "TotalSulfurDioxide", "Density", "PH", "Sulphates", "Alcohol"));
            pipeline.Add(new FastTreeRegressor());
            var model = pipeline.Train<WineData, WinePrediction>();
            return model;
        }

        static void Evaluate(PredictionModel<WineData, WinePrediction> model)
        {
            var testData = new Microsoft.ML.Data.TextLoader(TestDataPath).CreateFrom<WineData>(useHeader: true, separator: ',', trimWhitespace: false);
            var evaluator = new Microsoft.ML.Models.RegressionEvaluator();
            var metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine("Rms=" + metrics.Rms);
            Console.WriteLine("LossFn=" + metrics.LossFn);
            Console.WriteLine("RSquared = " + metrics.RSquared);
        }

        static void Predict(PredictionModel<WineData, WinePrediction> model)
        {
            using (var environment = new TlcEnvironment())
            {
                var textLoader = new Microsoft.ML.Data.TextLoader(TestDataPath).CreateFrom<WineData>(useHeader: true, separator: ',', trimWhitespace: false);
                var experiment = environment.CreateExperiment();
                var output = textLoader.ApplyStep(null, experiment) as ILearningPipelineDataStep;

                experiment.Compile();
                textLoader.SetInput(environment, experiment);
                experiment.Run();
                var data = experiment.GetOutput(output.Data);
                var wineDatas = new List<WineData>();
                using (var cursor = data.GetRowCursor((a => true)))
                {
                    var getters = new ValueGetter<float>[]{
                        cursor.GetGetter<float>(0),
                        cursor.GetGetter<float>(1),
                        cursor.GetGetter<float>(2),
                        cursor.GetGetter<float>(3),
                        cursor.GetGetter<float>(4),
                        cursor.GetGetter<float>(5),
                        cursor.GetGetter<float>(6),
                        cursor.GetGetter<float>(7),
                        cursor.GetGetter<float>(8),
                        cursor.GetGetter<float>(9),
                        cursor.GetGetter<float>(10),
                        cursor.GetGetter<float>(11),
                        cursor.GetGetter<float>(12)
                    };

                    while (cursor.MoveNext())
                    {
                        float value0 = 0;
                        float value1 = 0;
                        float value2 = 0;
                        float value3 = 0;
                        float value4 = 0;
                        float value5 = 0;
                        float value6 = 0;
                        float value7 = 0;
                        float value8 = 0;
                        float value9 = 0;
                        float value10 = 0;
                        float value11 = 0;
                        float value12 = 0;
                        getters[0](ref value0);
                        getters[1](ref value1);
                        getters[2](ref value2);
                        getters[3](ref value3);
                        getters[4](ref value4);
                        getters[5](ref value5);
                        getters[6](ref value6);
                        getters[7](ref value7);
                        getters[8](ref value8);
                        getters[9](ref value9);
                        getters[10](ref value10);
                        getters[11](ref value11);
                        getters[12](ref value12);

                        var wdata = new WineData()
                        {
                            FixedAcidity = value0,
                            VolatileAcidity = value1,
                            CitricACID = value2,
                            ResidualSugar = value3,
                            Chlorides = value4,
                            FreeSulfurDioxide = value5,
                            TotalSulfurDioxide = value6,
                            Density = value7,
                            PH = value8,
                            Sulphates = value9,
                            Alcohol = value10,
                            Quality = value11,
                            Id = value12,
                        };
                        wineDatas.Add(wdata);
                    }
                }
                var predictions = model.Predict(wineDatas);

                var wineDataAndPredictions = wineDatas.Zip(predictions, (wineData, prediction) => (wineData, prediction));
                Console.WriteLine($"Wine Id: {wineDataAndPredictions.Last().wineData.Id}, Quality: {wineDataAndPredictions.Last().wineData.Quality} | Prediction: {  wineDataAndPredictions.Last().prediction.PredictionQuality}");
                Console.WriteLine();
            }
        }

        static void Main(string[] args)
        {
            var model = Train();
            Evaluate(model);
            Predict(model);
            Console.ReadLine();
        }
    }
}