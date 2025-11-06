/**
 * ML Management Page - Example React Component
 *
 * This is an example implementation for the frontend.
 * Place in: frontend/src/pages/MLManagement/MLManagementPage.tsx
 */

import React, { useState, useEffect } from 'react';
import { Card, Button, Form, Progress, Table, Tag, Space, notification, Modal } from 'antd';
import {
  RocketOutlined,
  ReloadOutlined,
  CloudUploadOutlined,
  DownloadOutlined,
  DeleteOutlined,
  BarChartOutlined
} from '@ant-design/icons';

// Types
interface TrainingParams {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  export_onnx: boolean;
  auto_promote: boolean;
}

interface TrainingStatus {
  is_training: boolean;
  current_job?: {
    job_id: string;
    status: string;
    started_at: string;
    progress: {
      current_epoch: number;
      total_epochs: number;
      best_val_accuracy: number;
    };
  };
}

interface Model {
  name: string;
  version: string;
  stage: string;
  created_at: string;
  metrics: {
    accuracy?: number;
    precision?: number;
    recall?: number;
  };
}

const MLManagementPage: React.FC = () => {
  // State
  const [trainingParams, setTrainingParams] = useState<TrainingParams>({
    epochs: 50,
    batch_size: 64,
    learning_rate: 0.001,
    export_onnx: true,
    auto_promote: true
  });

  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    is_training: false
  });

  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(false);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);

  // Fetch training status
  const fetchTrainingStatus = async () => {
    try {
      const response = await fetch('/api/ml-management/training/status');
      const data = await response.json();
      setTrainingStatus(data);

      // If training is complete, show notification
      if (!data.is_training && pollingInterval) {
        clearInterval(pollingInterval);
        setPollingInterval(null);

        notification.success({
          message: 'Training Completed',
          description: 'Model training has finished successfully'
        });

        // Refresh models list
        fetchModels();
      }
    } catch (error) {
      console.error('Failed to fetch training status:', error);
    }
  };

  // Fetch models list
  const fetchModels = async () => {
    try {
      const response = await fetch('/api/ml-management/models');
      const data = await response.json();
      setModels(data.models);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  // Start training
  const handleStartTraining = async () => {
    try {
      setLoading(true);

      const response = await fetch('/api/ml-management/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainingParams)
      });

      const result = await response.json();

      if (result.job_id) {
        notification.success({
          message: 'Training Started',
          description: `Training job ${result.job_id} has been started`
        });

        // Start polling for status
        const interval = setInterval(fetchTrainingStatus, 2000);
        setPollingInterval(interval);
      }
    } catch (error) {
      notification.error({
        message: 'Training Failed',
        description: String(error)
      });
    } finally {
      setLoading(false);
    }
  };

  // Promote model
  const handlePromoteModel = async (name: string, version: string) => {
    Modal.confirm({
      title: 'Promote Model to Production',
      content: `Are you sure you want to promote ${name} v${version} to production?`,
      okText: 'Promote',
      okType: 'primary',
      onOk: async () => {
        try {
          const response = await fetch(
            `/api/ml-management/models/${name}/${version}/promote?stage=production`,
            { method: 'POST' }
          );

          const result = await response.json();

          if (result.success) {
            notification.success({
              message: 'Model Promoted',
              description: `${name} v${version} has been promoted to production`
            });
            fetchModels();
          }
        } catch (error) {
          notification.error({
            message: 'Promotion Failed',
            description: String(error)
          });
        }
      }
    });
  };

  // Initial fetch
  useEffect(() => {
    fetchTrainingStatus();
    fetchModels();

    // Cleanup
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, []);

  // Render training status card
  const renderTrainingStatus = () => {
    const { is_training, current_job } = trainingStatus;

    if (!is_training) {
      return (
        <Card title="📊 Training Status" style={{ marginBottom: 24 }}>
          <div style={{ textAlign: 'center', padding: '20px 0' }}>
            <p style={{ fontSize: 16, color: '#888' }}>No training in progress</p>
          </div>
        </Card>
      );
    }

    const progress = current_job
      ? (current_job.progress.current_epoch / current_job.progress.total_epochs) * 100
      : 0;

    return (
      <Card title="📊 Training Status" style={{ marginBottom: 24 }}>
        <div>
          <p>
            <strong>Status:</strong>{' '}
            <Tag color="processing">Training in progress</Tag>
          </p>
          <p>
            <strong>Job ID:</strong> {current_job?.job_id}
          </p>
          <p>
            <strong>Started:</strong> {current_job?.started_at}
          </p>

          <div style={{ marginTop: 16 }}>
            <p>
              <strong>
                Epoch: {current_job?.progress.current_epoch} /{' '}
                {current_job?.progress.total_epochs}
              </strong>
            </p>
            <Progress percent={Math.round(progress)} status="active" />
          </div>

          <div style={{ marginTop: 16 }}>
            <p>
              <strong>Best Val Accuracy:</strong>{' '}
              {current_job?.progress.best_val_accuracy.toFixed(4)}
            </p>
          </div>
        </div>
      </Card>
    );
  };

  // Render training form
  const renderTrainingForm = () => (
    <Card title="🎯 Quick Train" style={{ marginBottom: 24 }}>
      <Form layout="vertical">
        <Form.Item label="Epochs">
          <input
            type="number"
            className="ant-input"
            value={trainingParams.epochs}
            onChange={e =>
              setTrainingParams({
                ...trainingParams,
                epochs: parseInt(e.target.value)
              })
            }
            disabled={trainingStatus.is_training}
          />
        </Form.Item>

        <Form.Item label="Batch Size">
          <input
            type="number"
            className="ant-input"
            value={trainingParams.batch_size}
            onChange={e =>
              setTrainingParams({
                ...trainingParams,
                batch_size: parseInt(e.target.value)
              })
            }
            disabled={trainingStatus.is_training}
          />
        </Form.Item>

        <Form.Item label="Learning Rate">
          <input
            type="number"
            className="ant-input"
            step="0.0001"
            value={trainingParams.learning_rate}
            onChange={e =>
              setTrainingParams({
                ...trainingParams,
                learning_rate: parseFloat(e.target.value)
              })
            }
            disabled={trainingStatus.is_training}
          />
        </Form.Item>

        <Form.Item>
          <label>
            <input
              type="checkbox"
              checked={trainingParams.export_onnx}
              onChange={e =>
                setTrainingParams({
                  ...trainingParams,
                  export_onnx: e.target.checked
                })
              }
              disabled={trainingStatus.is_training}
            />
            {' Export to ONNX'}
          </label>
        </Form.Item>

        <Form.Item>
          <label>
            <input
              type="checkbox"
              checked={trainingParams.auto_promote}
              onChange={e =>
                setTrainingParams({
                  ...trainingParams,
                  auto_promote: e.target.checked
                })
              }
              disabled={trainingStatus.is_training}
            />
            {' Auto-promote to production'}
          </label>
        </Form.Item>

        <Button
          type="primary"
          icon={<RocketOutlined />}
          size="large"
          block
          onClick={handleStartTraining}
          loading={loading}
          disabled={trainingStatus.is_training}
        >
          {trainingStatus.is_training ? 'Training...' : 'Start Training'}
        </Button>
      </Form>
    </Card>
  );

  // Render models table
  const renderModelsTable = () => {
    const columns = [
      {
        title: 'Name',
        dataIndex: 'name',
        key: 'name'
      },
      {
        title: 'Version',
        dataIndex: 'version',
        key: 'version'
      },
      {
        title: 'Stage',
        dataIndex: 'stage',
        key: 'stage',
        render: (stage: string) => {
          const color =
            stage === 'production' ? 'green' : stage === 'staging' ? 'orange' : 'default';
          return <Tag color={color}>{stage.toUpperCase()}</Tag>;
        }
      },
      {
        title: 'Accuracy',
        key: 'accuracy',
        render: (record: Model) => record.metrics.accuracy?.toFixed(4) || 'N/A'
      },
      {
        title: 'Created',
        dataIndex: 'created_at',
        key: 'created_at',
        render: (date: string) => new Date(date).toLocaleString()
      },
      {
        title: 'Actions',
        key: 'actions',
        render: (record: Model) => (
          <Space>
            {record.stage !== 'production' && (
              <Button
                type="link"
                icon={<CloudUploadOutlined />}
                onClick={() => handlePromoteModel(record.name, record.version)}
              >
                Promote
              </Button>
            )}
            <Button type="link" icon={<DownloadOutlined />}>
              Download
            </Button>
            <Button type="link" icon={<BarChartOutlined />}>
              Metrics
            </Button>
          </Space>
        )
      }
    ];

    return (
      <Card
        title="📦 Models"
        extra={
          <Button icon={<ReloadOutlined />} onClick={fetchModels}>
            Refresh
          </Button>
        }
      >
        <Table
          dataSource={models}
          columns={columns}
          rowKey={record => `${record.name}_${record.version}`}
          pagination={{ pageSize: 10 }}
        />
      </Card>
    );
  };

  return (
    <div style={{ padding: 24 }}>
      <h1>ML Model Management</h1>

      {renderTrainingStatus()}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 24 }}>
        {renderTrainingForm()}
        <div>{renderModelsTable()}</div>
      </div>
    </div>
  );
};

export default MLManagementPage;
