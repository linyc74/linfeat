import pandas as pd
from test.setup import TestCase
from linfeat.basic import Parameters
from linfeat.workflows import statistics_workflow, feature_selection_workflow, feature_correlation_workflow


class TestStatisticsWorkflow(TestCase):

    def setUp(self):
        self.set_up(py_path=__file__)
        
    def tearDown(self):
        self.tear_down()
    
    def test_continuous_outcome(self):
        statistics_workflow(
            df=pd.read_csv(f'{self.indir}/data.csv', index_col=0),
            variables=[
                'Muribaculum_intestinale',
                'Ligilactobacillus',
                'Thomasclavelia',
                'Lachnospiraceae',
                'Faecalibaculum_rodentium',
                'Blautia',
                'Lachnoclostridium',
                'Erysipelotrichaceae',
                'Lactobacillus_johnsonii',
                'Bacteroides',
            ],
            outcome='Parabacteroides_goldsteinii',
            outdir=self.outdir,
            parametric_outcome=True,
            parametric_variables=[
                'Muribaculum_intestinale',
                'Ligilactobacillus',
                'Thomasclavelia',
                'Lachnospiraceae',
            ],
            colors='Set2',
        )
    
    def test_binary_outcome(self):
        statistics_workflow(
            df=pd.read_csv(f'{self.indir}/data.csv', index_col=0),
            variables=[
                'Muribaculum_intestinale',
                'Ligilactobacillus',
                'Thomasclavelia',
                'Lachnospiraceae',
                'Faecalibaculum_rodentium',
                'Blautia',
                'Lachnoclostridium',
                'Erysipelotrichaceae',
                'Lactobacillus_johnsonii',
                'Bacteroides',
            ],
            outcome='Obesity(1)/Normal(0)',
            outdir=self.outdir,
            parametric_outcome=False,
            parametric_variables=[
                'Muribaculum_intestinale',
                'Ligilactobacillus',
                'Thomasclavelia',
                'Lachnospiraceae',
            ],
            colors=['lightblue', 'tomato'],
        )


class TestFeatureCorrelationWorkflow(TestCase):

    def setUp(self):
        self.set_up(py_path=__file__)
        
    def tearDown(self):
        self.tear_down()
    
    def test_main(self):
        df = pd.read_csv(f'{self.indir}/data.csv', index_col=0)[[
            'Muribaculum_intestinale',
            'Ligilactobacillus',
            'Thomasclavelia',
            'Lachnospiraceae',
            'Faecalibaculum_rodentium',
            'Blautia',
            'Lachnoclostridium',
            'Erysipelotrichaceae',
            'Lactobacillus_johnsonii',
            'Bacteroides',
        ]]
        feature_correlation_workflow(df=df, outdir=self.outdir)


class TestFeatureSelectionWorkflow(TestCase):

    def setUp(self):
        self.set_up(py_path=__file__)
        
    def tearDown(self):
        self.tear_down()
    
    def test_numeric_outcome(self):
        parameters = Parameters()
        parameters.outdir = self.outdir
        feature_selection_workflow(
            df=pd.read_csv(f'{self.indir}/data.csv', index_col=0),
            features=[
                'Muribaculum_intestinale',
                'Ligilactobacillus',
                'Thomasclavelia',
                'Lachnospiraceae',
                'Faecalibaculum_rodentium',
                'Blautia',
                'Lachnoclostridium',
                'Erysipelotrichaceae',
                'Lactobacillus_johnsonii',
                'Bacteroides',
            ],
            outcome='Parabacteroides_goldsteinii',
            parameters=parameters,
        )

    def test_binary_outcome(self):
        parameters = Parameters()
        parameters.outdir = self.outdir
        feature_selection_workflow(
            df=pd.read_csv(f'{self.indir}/data.csv', index_col=0),
            features=[
                'Parabacteroides_goldsteinii',
                'Muribaculum_intestinale',
                'Ligilactobacillus',
                'Thomasclavelia',
                'Lachnospiraceae',
                'Faecalibaculum_rodentium',
                'Blautia',
                'Lachnoclostridium',
                'Erysipelotrichaceae',
                'Lactobacillus_johnsonii',
                'Bacteroides',
            ],
            outcome='Obesity(1)/Normal(0)',
            parameters=parameters,
        )
