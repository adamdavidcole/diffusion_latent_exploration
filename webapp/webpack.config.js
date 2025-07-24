const path = require('path');

module.exports = {
  entry: {
    app: './frontend/static/js/app.js',
    controls: './frontend/static/js/controls.js'
  },
  output: {
    path: path.resolve(__dirname, 'frontend/static/dist'),
    filename: '[name].min.js',
    clean: true
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env']
          }
        }
      }
    ]
  },
  optimization: {
    splitChunks: {
      chunks: 'all',
      cacheGroups: {
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendors',
          chunks: 'all'
        }
      }
    }
  }
};
