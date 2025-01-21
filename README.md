# ml-algo-trading

## HFT Engine Components

The HFT (High-Frequency Trading) Engine is designed to process market data, generate trading signals, manage orders, and ensure risk compliance efficiently. Below is an overview of its core components:

```
    HFT Engine
    ├── Market Data Handler
    │   └── Connect to data feed
    │   └── Process incoming data
    │   └── Update market state
    ├── Strategy Engine
    │   └── Generate trading signals
    │   └── Send signals to order manager
    ├── Order Manager
    │   └── Place orders
    │   └── Track order status
    │   └── Handle order events
    ├── Risk Manager
        └── Monitor trading limits
        └── Validate orders before execution
```