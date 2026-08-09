import React, { useState, useEffect } from 'react';
import { Card, Button, Form, Input, Select, Slider, Spin, Alert, Tabs, Table, Space, Typography, Upload, message } from 'antd';
import { UploadOutlined, CloudOutlined, ThunderboltOutlined, AreaChartOutlined } from '@ant-design/icons';
import axios from 'axios';
import { Line, Bar } from 'react-chartjs-2';
import { API_BASE_URL } from '../../services/api';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const ClimateResilienceProcessor = () => {
  const [loading, setLoading] = useState(false);
  const [climateData, setClimateData] = useState(null);
  const [explorationArea, setExplorationArea] = useState(null);
  const [weatherRiskResults, setWeatherRiskResults] = useState(null);
  const [waterResourceResults, setWaterResourceResults] = useState(null);
  const [operationalResults, setOperationalResults] = useState(null);
  const [carbonResults, setCarbonResults] = useState(null);
  const [reportUrl, setReportUrl] = useState(null);
  const [activeTab, setActiveTab] = useState('1');

  // Form states
  const [weatherThresholds, setWeatherThresholds] = useState({
    heavy_precipitation: 50,
    extreme_heat: 35,
    drought: 30,
    high_wind: 80
  });

  const [waterUsage, setWaterUsage] = useState({
    processing: 5000,
    dust_suppression: 1000,
    potable: 500
  });

  const [operationalParams, setOperationalParams] = useState({
    precipitation_threshold: 50,
    temperature_threshold: 35,
    daily_operation_cost: 100000,
    daily_revenue: 150000
  });

  const [adaptationOptions, setAdaptationOptions] = useState([
    {
      name: 'Weather resistant equipment',
      disruption_reduction_factor: 0.5,
      implementation_cost: 1000000,
      annual_maintenance: 100000,
      lifespan: 10
    },
    {
      name: 'Water recycling system',
      disruption_reduction_factor: 0.3,
      implementation_cost: 2000000,
      annual_maintenance: 200000,
      lifespan: 15
    }
  ]);

  const [carbonData, setCarbonData] = useState({
    fuel_consumption: 500000, // liters/year
    electricity_usage: 10000000, // kWh/year
    process_emissions: 20000, // tonnes CO2e/year
    employee_travel: 5000 // tonnes CO2e/year
  });

  useEffect(() => {
    // Load exploration area data if available
    const loadExplorationArea = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/exploration-areas/current`);
        if (response.data) {
          setExplorationArea(response.data);
        }
      } catch (error) {
        console.error('Error loading exploration area:', error);
      }
    };

    loadExplorationArea();
  }, []);

  const handleClimateDataUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('data_type', 'precipitation');
    
    if (explorationArea) {
      formData.append('region_min_lon', explorationArea.region.min_lon);
      formData.append('region_max_lon', explorationArea.region.max_lon);
      formData.append('region_min_lat', explorationArea.region.min_lat);
      formData.append('region_max_lat', explorationArea.region.max_lat);
    } else {
      formData.append('region_min_lon', -120.5);
      formData.append('region_max_lon', -120.0);
      formData.append('region_min_lat', 38.0);
      formData.append('region_max_lat', 38.5);
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/climate-resilience/upload-climate-data`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setClimateData(response.data);
      message.success('Climate data uploaded successfully');
    } catch (error) {
      console.error('Error uploading climate data:', error);
      message.error('Failed to upload climate data');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadClimateData = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/climate-resilience/load-climate-data`, {
        data_type: 'precipitation',
        source: 'worldclim',
        region: explorationArea ? explorationArea.region : {
          min_lon: -120.5,
          max_lon: -120.0,
          min_lat: 38.0,
          max_lat: 38.5
        },
        time_range: ['2020-01-01', '2050-12-31']
      });
      setClimateData(response.data);
      message.success('Climate data loaded successfully');
    } catch (error) {
      console.error('Error loading climate data:', error);
      message.error('Failed to load climate data');
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeExtremeWeather = async () => {
    if (!climateData) {
      message.warning('Please load climate data first');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/climate-resilience/analyze-extreme-weather`, {
        exploration_area: explorationArea || {
          name: 'Default Exploration Area',
          region: {
            min_lon: -120.5,
            max_lon: -120.0,
            min_lat: 38.0,
            max_lat: 38.5
          }
        },
        thresholds: weatherThresholds
      });
      setWeatherRiskResults(response.data);
      setActiveTab('2');
      message.success('Extreme weather risk analysis completed');
    } catch (error) {
      console.error('Error analyzing extreme weather risk:', error);
      message.error('Failed to analyze extreme weather risk');
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeWaterResources = async () => {
    if (!climateData) {
      message.warning('Please load climate data first');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/climate-resilience/analyze-water-resources`, {
        exploration_area: explorationArea || {
          name: 'Default Exploration Area',
          region: {
            min_lon: -120.5,
            max_lon: -120.0,
            min_lat: 38.0,
            max_lat: 38.5
          }
        },
        water_usage: waterUsage
      });
      setWaterResourceResults(response.data);
      setActiveTab('3');
      message.success('Water resource impact analysis completed');
    } catch (error) {
      console.error('Error analyzing water resource impacts:', error);
      message.error('Failed to analyze water resource impacts');
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeOperationalResilience = async () => {
    if (!climateData) {
      message.warning('Please load climate data first');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/climate-resilience/analyze-operational-resilience`, {
        exploration_area: explorationArea || {
          name: 'Default Exploration Area',
          region: {
            min_lon: -120.5,
            max_lon: -120.0,
            min_lat: 38.0,
            max_lat: 38.5
          }
        },
        operational_params: {
          ...operationalParams,
          adaptation_options: adaptationOptions.reduce((acc, option) => {
            acc[option.name.replace(/\s+/g, '_').toLowerCase()] = {
              disruption_reduction_factor: option.disruption_reduction_factor,
              implementation_cost: option.implementation_cost,
              annual_maintenance: option.annual_maintenance,
              lifespan: option.lifespan
            };
            return acc;
          }, {})
        }
      });
      setOperationalResults(response.data);
      setActiveTab('4');
      message.success('Operational resilience analysis completed');
    } catch (error) {
      console.error('Error analyzing operational resilience:', error);
      message.error('Failed to analyze operational resilience');
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeCarbonFootprint = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/climate-resilience/analyze-carbon-footprint`, {
        operational_data: carbonData,
        reduction_scenarios: [
          {
            name: 'Renewable Energy Implementation',
            reductions: {
              electricity_usage: 0.7 // 70% reduction in emissions from electricity
            },
            implementation_cost: 5000000,
            annual_savings: 500000,
            payback_period: 10
          },
          {
            name: 'Electric Vehicle Fleet',
            reductions: {
              fuel_consumption: 0.8 // 80% reduction in emissions from fuel
            },
            implementation_cost: 3000000,
            annual_savings: 400000,
            payback_period: 7.5
          }
        ]
      });
      setCarbonResults(response.data);
      setActiveTab('5');
      message.success('Carbon footprint analysis completed');
    } catch (error) {
      console.error('Error analyzing carbon footprint:', error);
      message.error('Failed to analyze carbon footprint');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!weatherRiskResults || !waterResourceResults || !operationalResults) {
      message.warning('Please complete all analyses first');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/climate-resilience/generate-report`, {
        exploration_area: explorationArea || {
          name: 'Default Exploration Area',
          region: {
            min_lon: -120.5,
            max_lon: -120.0,
            min_lat: 38.0,
            max_lat: 38.5
          }
        },
        analyses: {
          extreme_weather_risk: weatherRiskResults,
          water_resource_impacts: waterResourceResults,
          operational_resilience: operationalResults,
          carbon_footprint: carbonResults
        }
      });
      setReportUrl(response.data.report_file);
      setActiveTab('6');
      message.success('Climate resilience report generated successfully');
    } catch (error) {
      console.error('Error generating climate resilience report:', error);
      message.error('Failed to generate climate resilience report');
    } finally {
      setLoading(false);
    }
  };

  const renderDataUploadTab = () => (
    <div style={{ padding: '20px 0' }}>
      <Card title="Climate Data Management" style={{ marginBottom: 20 }}>
        <Paragraph>
          Upload climate data files or load data from global climate databases to analyze climate resilience for your exploration area.
        </Paragraph>
        
        <div style={{ display: 'flex', marginBottom: 20 }}>
          <Upload
            beforeUpload={(file) => {
              handleClimateDataUpload(file);
              return false;
            }}
            showUploadList={false}
          >
            <Button icon={<UploadOutlined />}>Upload Climate Data</Button>
          </Upload>
          
          <Button 
            type="primary" 
            icon={<CloudOutlined />} 
            onClick={handleLoadClimateData}
            style={{ marginLeft: 10 }}
          >
            Load Global Climate Data
          </Button>
        </div>
        
        {climateData && (
          <Alert
            message="Climate Data Loaded"
            description={`Successfully loaded ${climateData.data_type} data from ${climateData.source} for the region ${JSON.stringify(climateData.region)}`}
            type="success"
            showIcon
          />
        )}
      </Card>
      
      <Card title="Extreme Weather Risk Analysis" style={{ marginBottom: 20 }}>
        <Paragraph>
          Analyze the risk of extreme weather events for your exploration area based on climate projections.
        </Paragraph>
        
        <Form layout="vertical">
          <Form.Item label="Heavy Precipitation Threshold (mm/day)">
            <Slider
              min={10}
              max={200}
              value={weatherThresholds.heavy_precipitation}
              onChange={(value) => setWeatherThresholds({...weatherThresholds, heavy_precipitation: value})}
            />
          </Form.Item>
          
          <Form.Item label="Extreme Heat Threshold (°C)">
            <Slider
              min={25}
              max={50}
              value={weatherThresholds.extreme_heat}
              onChange={(value) => setWeatherThresholds({...weatherThresholds, extreme_heat: value})}
            />
          </Form.Item>
          
          <Form.Item>
            <Button 
              type="primary" 
              onClick={handleAnalyzeExtremeWeather}
              disabled={!climateData}
            >
              Analyze Extreme Weather Risk
            </Button>
          </Form.Item>
        </Form>
      </Card>
      
      <Card title="Water Resource Analysis" style={{ marginBottom: 20 }}>
        <Paragraph>
          Analyze the impacts of climate change on water resources for your mining operations.
        </Paragraph>
        
        <Form layout="vertical">
          <Form.Item label="Processing Water Usage (m³/day)">
            <Slider
              min={1000}
              max={10000}
              step={100}
              value={waterUsage.processing}
              onChange={(value) => setWaterUsage({...waterUsage, processing: value})}
            />
          </Form.Item>
          
          <Form.Item label="Dust Suppression Water Usage (m³/day)">
            <Slider
              min={100}
              max={5000}
              step={100}
              value={waterUsage.dust_suppression}
              onChange={(value) => setWaterUsage({...waterUsage, dust_suppression: value})}
            />
          </Form.Item>
          
          <Form.Item label="Potable Water Usage (m³/day)">
            <Slider
              min={100}
              max={1000}
              step={50}
              value={waterUsage.potable}
              onChange={(value) => setWaterUsage({...waterUsage, potable: value})}
            />
          </Form.Item>
          
          <Form.Item>
            <Button 
              type="primary" 
              onClick={handleAnalyzeWaterResources}
              disabled={!climateData}
            >
              Analyze Water Resource Impacts
            </Button>
          </Form.Item>
        </Form>
      </Card>
      
      <Card title="Operational Resilience Analysis">
        <Paragraph>
          Analyze the resilience of mining operations to climate impacts and evaluate adaptation options.
        </Paragraph>
        
        <Form layout="vertical">
          <Form.Item label="Precipitation Disruption Threshold (mm/day)">
            <Slider
              min={10}
              max={100}
              value={operationalParams.precipitation_threshold}
              onChange={(value) => setOperationalParams({...operationalParams, precipitation_threshold: value})}
            />
          </Form.Item>
          
          <Form.Item label="Temperature Disruption Threshold (°C)">
            <Slider
              min={25}
              max={45}
              value={operationalParams.temperature_threshold}
              onChange={(value) => setOperationalParams({...operationalParams, temperature_threshold: value})}
            />
          </Form.Item>
          
          <Form.Item label="Daily Operation Cost ($)">
            <Input
              type="number"
              value={operationalParams.daily_operation_cost}
              onChange={(e) => setOperationalParams({...operationalParams, daily_operation_cost: Number(e.target.value)})}
            />
          </Form.Item>
          
          <Form.Item label="Daily Revenue ($)">
            <Input
              type="number"
              value={operationalParams.daily_revenue}
              onChange={(e) => setOperationalParams({...operationalParams, daily_revenue: Number(e.target.value)})}
            />
          </Form.Item>
          
          <Form.Item>
            <Button 
              type="primary" 
              onClick={handleAnalyzeOperationalResilience}
              disabled={!climateData}
            >
              Analyze Operational Resilience
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );

  const renderWeatherRiskTab = () => (
    <div style={{ padding: '20px 0' }}>
      {weatherRiskResults ? (
        <>
          <Card title="Extreme Weather Risk Analysis Results" style={{ marginBottom: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <div style={{ width: '48%' }}>
                <Title level={4}>Heavy Precipitation Risk</Title>
                <Paragraph>
                  <Text strong>Annual Frequency: </Text> 
                  {weatherRiskResults.heavy_precipitation.annual_frequency} days/year
                </Paragraph>
                <Paragraph>
                  <Text strong>Trend: </Text> 
                  {weatherRiskResults.heavy_precipitation.trend > 0 ? 'Increasing' : 'Decreasing'} 
                  ({Math.abs(weatherRiskResults.heavy_precipitation.trend * 100).toFixed(1)}% per decade)
                </Paragraph>
                <Paragraph>
                  <Text strong>Risk Level: </Text> 
                  <Text type={weatherRiskResults.heavy_precipitation.risk_level === 'High' ? 'danger' : 
                             weatherRiskResults.heavy_precipitation.risk_level === 'Medium' ? 'warning' : 'success'}>
                    {weatherRiskResults.heavy_precipitation.risk_level}
                  </Text>
                </Paragraph>
              </div>
              
              <div style={{ width: '48%' }}>
                <Title level={4}>Extreme Heat Risk</Title>
                <Paragraph>
                  <Text strong>Annual Frequency: </Text> 
                  {weatherRiskResults.extreme_heat.annual_frequency} days/year
                </Paragraph>
                <Paragraph>
                  <Text strong>Trend: </Text> 
                  {weatherRiskResults.extreme_heat.trend > 0 ? 'Increasing' : 'Decreasing'} 
                  ({Math.abs(weatherRiskResults.extreme_heat.trend * 100).toFixed(1)}% per decade)
                </Paragraph>
                <Paragraph>
                  <Text strong>Risk Level: </Text> 
                  <Text type={weatherRiskResults.extreme_heat.risk_level === 'High' ? 'danger' : 
                             weatherRiskResults.extreme_heat.risk_level === 'Medium' ? 'warning' : 'success'}>
                    {weatherRiskResults.extreme_heat.risk_level}
                  </Text>
                </Paragraph>
              </div>
            </div>
            
            {/* Placeholder for chart - in a real implementation, this would use actual data */}
            <div style={{ height: 300, marginTop: 20 }}>
              {/* <Line
                data={{
                  labels: ['2025', '2030', '2035', '2040', '2045', '2050'],
                  datasets: [
                    {
                      label: 'Heavy Precipitation Days',
                      data: [5, 6, 7, 8, 9, 10],
                      borderColor: 'rgba(54, 162, 235, 1)',
                      backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    },
                    {
                      label: 'Extreme Heat Days',
                      data: [10, 12, 15, 18, 20, 25],
                      borderColor: 'rgba(255, 99, 132, 1)',
                      backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    }
                  ]
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      beginAtZero: true,
                      title: {
                        display: true,
                        text: 'Days per Year'
                      }
                    },
                    x: {
                      title: {
                        display: true,
                        text: 'Year'
                      }
                    }
                  }
                }}
              /> */}
              <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                <Text type="secondary">Chart visualization would appear here with actual data</Text>
              </div>
            </div>
          </Card>
          
          <Card title="Recommendations">
            <Title level={4}>Mitigation Strategies</Title>
            <ul>
              {weatherRiskResults.recommendations.map((rec, index) => (
                <li key={index}><Paragraph>{rec}</Paragraph></li>
              ))}
            </ul>
            
            <Button type="primary" onClick={() => setActiveTab('3')}>
              Continue to Water Resource Analysis
            </Button>
          </Card>
        </>
      ) : (
        <Alert
          message="No Analysis Results"
          description="Please complete the extreme weather risk analysis first."
          type="info"
          showIcon
        />
      )}
    </div>
  );

  const renderWaterResourceTab = () => (
    <div style={{ padding: '20px 0' }}>
      {waterResourceResults ? (
        <>
          <Card title="Water Resource Impact Analysis Results" style={{ marginBottom: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <div style={{ width: '48%' }}>
                <Title level={4}>Precipitation Projections</Title>
                <Paragraph>
                  <Text strong>Current Annual Precipitation: </Text> 
                  {waterResourceResults.precipitation.current} mm/year
                </Paragraph>
                <Paragraph>
                  <Text strong>Projected Change: </Text> 
                  {waterResourceResults.precipitation.projected_change > 0 ? '+' : ''}
                  {waterResourceResults.precipitation.projected_change}% by 2050
                </Paragraph>
                <Paragraph>
                  <Text strong>Seasonal Variation: </Text> 
                  {waterResourceResults.precipitation.seasonal_variation}
                </Paragraph>
              </div>
              
              <div style={{ width: '48%' }}>
                <Title level={4}>Water Stress Assessment</Title>
                <Paragraph>
                  <Text strong>Current Water Stress Level: </Text> 
                  <Text type={waterResourceResults.water_stress.current_level === 'High' ? 'danger' : 
                             waterResourceResults.water_stress.current_level === 'Medium' ? 'warning' : 'success'}>
                    {waterResourceResults.water_stress.current_level}
                  </Text>
                </Paragraph>
                <Paragraph>
                  <Text strong>Projected Water Stress Level: </Text> 
                  <Text type={waterResourceResults.water_stress.projected_level === 'High' ? 'danger' : 
                             waterResourceResults.water_stress.projected_level === 'Medium' ? 'warning' : 'success'}>
                    {waterResourceResults.water_stress.projected_level}
                  </Text>
                </Paragraph>
                <Paragraph>
                  <Text strong>Water Supply Risk: </Text> 
                  <Text type={waterResourceResults.water_stress.supply_risk === 'High' ? 'danger' : 
                             waterResourceResults.water_stress.supply_risk === 'Medium' ? 'warning' : 'success'}>
                    {waterResourceResults.water_stress.supply_risk}
                  </Text>
                </Paragraph>
              </div>
            </div>
            
            {/* Placeholder for chart - in a real implementation, this would use actual data */}
            <div style={{ height: 300, marginTop: 20 }}>
              {/* <Bar
                data={{
                  labels: ['Current', '2030', '2040', '2050'],
                  datasets: [
                    {
                      label: 'Water Availability (million m³)',
                      data: [10, 9, 8, 7],
                      backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    },
                    {
                      label: 'Water Demand (million m³)',
                      data: [6, 7, 8, 9],
                      backgroundColor: 'rgba(255, 99, 132, 0.5)',
                    }
                  ]
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      beginAtZero: true,
                      title: {
                        display: true,
                        text: 'Million m³'
                      }
                    }
                  }
                }}
              /> */}
              <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                <Text type="secondary">Chart visualization would appear here with actual data</Text>
              </div>
            </div>
          </Card>
          
          <Card title="Water Management Recommendations">
            <Title level={4}>Water Conservation Strategies</Title>
            <ul>
              {waterResourceResults.recommendations.map((rec, index) => (
                <li key={index}><Paragraph>{rec}</Paragraph></li>
              ))}
            </ul>
            
            <Button type="primary" onClick={() => setActiveTab('4')}>
              Continue to Operational Resilience Analysis
            </Button>
          </Card>
        </>
      ) : (
        <Alert
          message="No Analysis Results"
          description="Please complete the water resource impact analysis first."
          type="info"
          showIcon
        />
      )}
    </div>
  );

  const renderOperationalResilienceTab = () => (
    <div style={{ padding: '20px 0' }}>
      {operationalResults ? (
        <>
          <Card title="Operational Resilience Analysis Results" style={{ marginBottom: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <div style={{ width: '48%' }}>
                <Title level={4}>Operational Disruptions</Title>
                <Paragraph>
                  <Text strong>Current Annual Disruption Days: </Text> 
                  {operationalResults.operational_disruptions.current_annual_days} days/year
                </Paragraph>
                <Paragraph>
                  <Text strong>Projected Annual Disruption Days (2050): </Text> 
                  {operationalResults.operational_disruptions.projected_annual_days} days/year
                </Paragraph>
                <Paragraph>
                  <Text strong>Primary Disruption Cause: </Text> 
                  {operationalResults.operational_disruptions.primary_cause}
                </Paragraph>
              </div>
              
              <div style={{ width: '48%' }}>
                <Title level={4}>Financial Impacts</Title>
                <Paragraph>
                  <Text strong>Current Annual Loss: </Text> 
                  ${operationalResults.financial_impacts.current_annual_loss.toLocaleString()}
                </Paragraph>
                <Paragraph>
                  <Text strong>Projected Annual Loss (2050): </Text> 
                  ${operationalResults.financial_impacts.projected_annual_loss.toLocaleString()}
                </Paragraph>
                <Paragraph>
                  <Text strong>Net Present Value of Future Losses: </Text> 
                  ${operationalResults.financial_impacts.npv_future_losses.toLocaleString()}
                </Paragraph>
              </div>
            </div>
            
            <Title level={4} style={{ marginTop: 20 }}>Adaptation Options Analysis</Title>
            <Table
              dataSource={operationalResults.adaptation_options.map((option, index) => ({
                key: index,
                name: option.name,
                disruption_reduction: `${(option.disruption_reduction * 100).toFixed(0)}%`,
                implementation_cost: `$${option.implementation_cost.toLocaleString()}`,
                annual_savings: `$${option.annual_savings.toLocaleString()}`,
                payback_period: `${option.payback_period.toFixed(1)} years`,
                roi: `${(option.roi * 100).toFixed(1)}%`,
                recommendation: option.recommended ? 'Recommended' : 'Not Recommended'
              }))}
              columns={[
                { title: 'Adaptation Option', dataIndex: 'name', key: 'name' },
                { title: 'Disruption Reduction', dataIndex: 'disruption_reduction', key: 'disruption_reduction' },
                { title: 'Implementation Cost', dataIndex: 'implementation_cost', key: 'implementation_cost' },
                { title: 'Annual Savings', dataIndex: 'annual_savings', key: 'annual_savings' },
                { title: 'Payback Period', dataIndex: 'payback_period', key: 'payback_period' },
                { title: 'ROI', dataIndex: 'roi', key: 'roi' },
                { 
                  title: 'Recommendation', 
                  dataIndex: 'recommendation', 
                  key: 'recommendation',
                  render: (text) => (
                    <Text type={text === 'Recommended' ? 'success' : 'secondary'}>
                      {text}
                    </Text>
                  )
                }
              ]}
              pagination={false}
            />
          </Card>
          
          <Button type="primary" onClick={() => setActiveTab('5')}>
            Continue to Carbon Footprint Analysis
          </Button>
        </>
      ) : (
        <Alert
          message="No Analysis Results"
          description="Please complete the operational resilience analysis first."
          type="info"
          showIcon
        />
      )}
    </div>
  );

  const renderCarbonFootprintTab = () => (
    <div style={{ padding: '20px 0' }}>
      {carbonResults ? (
        <>
          <Card title="Carbon Footprint Analysis Results" style={{ marginBottom: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <div style={{ width: '48%' }}>
                <Title level={4}>Current Carbon Footprint</Title>
                <Paragraph>
                  <Text strong>Total Annual Emissions: </Text> 
                  {carbonResults.total_emissions.toLocaleString()} tonnes CO₂e/year
                </Paragraph>
                <Paragraph>
                  <Text strong>Emissions Intensity: </Text> 
                  {carbonResults.emissions_intensity.toFixed(2)} tonnes CO₂e/tonne of product
                </Paragraph>
                <Paragraph>
                  <Text strong>Largest Emission Source: </Text> 
                  {carbonResults.largest_source.name} ({carbonResults.largest_source.percentage.toFixed(1)}%)
                </Paragraph>
              </div>
              
              <div style={{ width: '48%' }}>
                <Title level={4}>Regulatory Compliance</Title>
                <Paragraph>
                  <Text strong>Carbon Price Exposure: </Text> 
                  ${carbonResults.carbon_price_exposure.toLocaleString()}/year
                </Paragraph>
                <Paragraph>
                  <Text strong>Compliance Status: </Text> 
                  <Text type={carbonResults.compliance_status === 'Compliant' ? 'success' : 'danger'}>
                    {carbonResults.compliance_status}
                  </Text>
                </Paragraph>
                <Paragraph>
                  <Text strong>Reporting Requirements: </Text> 
                  {carbonResults.reporting_requirements}
                </Paragraph>
              </div>
            </div>
            
            <Title level={4} style={{ marginTop: 20 }}>Emission Reduction Scenarios</Title>
            <Table
              dataSource={carbonResults.reduction_scenarios.map((scenario, index) => ({
                key: index,
                name: scenario.name,
                emission_reduction: `${(scenario.emission_reduction * 100).toFixed(0)}%`,
                implementation_cost: `$${scenario.implementation_cost.toLocaleString()}`,
                annual_savings: `$${scenario.annual_savings.toLocaleString()}`,
                payback_period: `${scenario.payback_period.toFixed(1)} years`,
                npv: `$${scenario.npv.toLocaleString()}`,
                recommendation: scenario.recommended ? 'Recommended' : 'Not Recommended'
              }))}
              columns={[
                { title: 'Reduction Scenario', dataIndex: 'name', key: 'name' },
                { title: 'Emission Reduction', dataIndex: 'emission_reduction', key: 'emission_reduction' },
                { title: 'Implementation Cost', dataIndex: 'implementation_cost', key: 'implementation_cost' },
                { title: 'Annual Savings', dataIndex: 'annual_savings', key: 'annual_savings' },
                { title: 'Payback Period', dataIndex: 'payback_period', key: 'payback_period' },
                { title: 'NPV', dataIndex: 'npv', key: 'npv' },
                { 
                  title: 'Recommendation', 
                  dataIndex: 'recommendation', 
                  key: 'recommendation',
                  render: (text) => (
                    <Text type={text === 'Recommended' ? 'success' : 'secondary'}>
                      {text}
                    </Text>
                  )
                }
              ]}
              pagination={false}
            />
          </Card>
          
          <Button type="primary" onClick={handleGenerateReport}>
            Generate Comprehensive Climate Resilience Report
          </Button>
        </>
      ) : (
        <>
          <Card title="Carbon Footprint Analysis" style={{ marginBottom: 20 }}>
            <Paragraph>
              Analyze the carbon footprint of mining operations and potential reduction strategies.
            </Paragraph>
            
            <Form layout="vertical">
              <Form.Item label="Fuel Consumption (liters/year)">
                <Input
                  type="number"
                  value={carbonData.fuel_consumption}
                  onChange={(e) => setCarbonData({...carbonData, fuel_consumption: Number(e.target.value)})}
                />
              </Form.Item>
              
              <Form.Item label="Electricity Usage (kWh/year)">
                <Input
                  type="number"
                  value={carbonData.electricity_usage}
                  onChange={(e) => setCarbonData({...carbonData, electricity_usage: Number(e.target.value)})}
                />
              </Form.Item>
              
              <Form.Item label="Process Emissions (tonnes CO₂e/year)">
                <Input
                  type="number"
                  value={carbonData.process_emissions}
                  onChange={(e) => setCarbonData({...carbonData, process_emissions: Number(e.target.value)})}
                />
              </Form.Item>
              
              <Form.Item label="Employee Travel (tonnes CO₂e/year)">
                <Input
                  type="number"
                  value={carbonData.employee_travel}
                  onChange={(e) => setCarbonData({...carbonData, employee_travel: Number(e.target.value)})}
                />
              </Form.Item>
              
              <Form.Item>
                <Button 
                  type="primary" 
                  onClick={handleAnalyzeCarbonFootprint}
                >
                  Analyze Carbon Footprint
                </Button>
              </Form.Item>
            </Form>
          </Card>
        </>
      )}
    </div>
  );

  const renderReportTab = () => (
    <div style={{ padding: '20px 0' }}>
      {reportUrl ? (
        <Card title="Climate Resilience Report">
          <Alert
            message="Report Generated Successfully"
            description="Your comprehensive climate resilience report has been generated successfully."
            type="success"
            showIcon
            style={{ marginBottom: 20 }}
          />
          
          <div style={{ textAlign: 'center', margin: '20px 0' }}>
            <Button type="primary" size="large" href={reportUrl} target="_blank">
              Download Report
            </Button>
          </div>
          
          <Title level={4}>Report Contents</Title>
          <ul>
            <li><Paragraph>Executive Summary</Paragraph></li>
            <li><Paragraph>Climate Risk Profile</Paragraph></li>
            <li><Paragraph>Extreme Weather Risk Analysis</Paragraph></li>
            <li><Paragraph>Water Resource Impact Assessment</Paragraph></li>
            <li><Paragraph>Operational Resilience Analysis</Paragraph></li>
            <li><Paragraph>Carbon Footprint Assessment</Paragraph></li>
            <li><Paragraph>Adaptation and Mitigation Recommendations</Paragraph></li>
            <li><Paragraph>Implementation Roadmap</Paragraph></li>
            <li><Paragraph>Financial Analysis</Paragraph></li>
            <li><Paragraph>Appendices and Supporting Data</Paragraph></li>
          </ul>
        </Card>
      ) : (
        <Alert
          message="No Report Generated"
          description="Please complete all analyses and generate a report first."
          type="info"
          showIcon
        />
      )}
    </div>
  );

  return (
    <div className="climate-resilience-processor">
      <Title level={2}>
        <ThunderboltOutlined /> Climate Resilience Analysis
      </Title>
      
      <Paragraph>
        Assess and enhance the climate resilience of exploration and mining operations with comprehensive analysis tools.
      </Paragraph>
      
      <Spin spinning={loading}>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="Data & Analysis" key="1">
            {renderDataUploadTab()}
          </TabPane>
          <TabPane tab="Weather Risk" key="2">
            {renderWeatherRiskTab()}
          </TabPane>
          <TabPane tab="Water Resources" key="3">
            {renderWaterResourceTab()}
          </TabPane>
          <TabPane tab="Operational Resilience" key="4">
            {renderOperationalResilienceTab()}
          </TabPane>
          <TabPane tab="Carbon Footprint" key="5">
            {renderCarbonFootprintTab()}
          </TabPane>
          <TabPane tab="Report" key="6">
            {renderReportTab()}
          </TabPane>
        </Tabs>
      </Spin>
    </div>
  );
};

export default ClimateResilienceProcessor;
