const app = getApp()
const API_BASE = 'https://192.168.81.208:5000' // 替换为实际域名
const hlqSceneData = require('../../data/HLQ_Scene_train.js')

Page({
  data: {
    allProducts: [],       // 从JSON加载的全部商品数据
    products: [],          // 全部商品数据
    displayedProducts: [], // 当前显示的商品
    searchKeyword: '',     // 搜索关键词
    page: 1,               // 当前页码
    pageSize: 20,          // 每页数量
    hasMore: true,         // 是否有更多数据
    loading: false,        // 加载状态
    selectedProduct: null, // 选中的商品
    searching: false,      // 是否正在搜索
    dataLoaded: false,     // 数据是否已加载
    
    // 谈判相关状态
    userId: '',            // 用户ID
    messages: [],          // 聊天消息
    messageId: 0,          // 消息ID计数器
    inputValue: '',        // 输入框内容
    negotiationStatus: 'not_started', // 谈判状态：not_started, ongoing, ended
    //finalOffer: null,      // 最终报价
    productInfo: null,     // 当前谈判商品信息
    scrollTop: 0,          // 滚动位置
    isNegotiating: false,  // 是否在谈判状态
    //pollTimer: null,       // 轮询定时器
    welcomeMessage: '' ,   // 欢迎消息
    isWaitingForBot: false, // 新增状态：防止用户重复发送消息
    inactivityTimer: null, // 【新增】用于客户端超时逻辑的定时器ID
  },

  onLoad() {
    this.generateUserId() // 生成用户ID
    this.loadProductsFromJson()// 加载产品数据
  },
  
  onUnload() {
    // 清除轮询定时器
    if (this.data.inactivityTimer) {
      clearTimeout(this.data.inactivityTimer);
      this.setData({ inactivityTimer: null });
    }
  },

  // 生成用户唯一ID
  generateUserId() {
    const storedUserId = wx.getStorageSync('user_id')
    if (storedUserId) {
      this.setData({ userId: storedUserId })
    } else {
      const newUserId = 'user_' + Date.now() + '_' + Math.floor(Math.random() * 10000)
      wx.setStorageSync('user_id', newUserId)
      this.setData({ userId: newUserId })
    }
  },

  loadProductsFromJson() {
    this.setData({ loading: true })
    try {
      // 直接使用导入的数据
      this.processJsonData(hlqSceneData)
    } catch (error) {
      console.error('数据加载错误:', error)
      this.setData({
        allProducts: [],
        products: [],
        displayedProducts: [],
        loading: false,
        dataLoaded: false,
        hasMore: false
      })
      
      wx.showToast({
        title: '数据加载失败',
        icon: 'error',
        duration: 5000
      })
    }
  },

  // 格式化价格显示
  formatPrice(price) {
    if (typeof price !== 'number') return '¥0'
    return `¥${price.toLocaleString()}`
  },

  // 处理JSON数据，转换为小程序需要的格式
  processJsonData(jsonData) {
    const transformedProducts = jsonData.scenes.map(item => ({
      id: item.product_id.toString(),
      name: item.product_name,
      description: item.seller_item_description,
      basePrice: item.seller_price,
      buyerReservePrice: item.buyer_reserve_price,
      sellerReservePrice: item.seller_reserve_price,
      image: item.image
    }))

    this.setData({
      allProducts: transformedProducts,
      products: transformedProducts,
      displayedProducts: transformedProducts.slice(0, this.data.pageSize),
      loading: false,
      dataLoaded: true,
      hasMore: transformedProducts.length > this.data.pageSize
    })
  },

  // 加载更多商品（分页）
  loadMoreProducts() {
    if (this.data.loading || !this.data.hasMore) return
    
    const startIndex = this.data.page * this.data.pageSize
    const endIndex = startIndex + this.data.pageSize
    const moreProducts = this.data.allProducts.slice(startIndex, endIndex)
    
    if (moreProducts.length > 0) {
      const newDisplayed = this.data.displayedProducts.concat(moreProducts)
      this.setData({
        displayedProducts: this.data.searching ? this.filterProducts(newDisplayed) : newDisplayed,
        page: this.data.page + 1,
        hasMore: endIndex < this.data.allProducts.length
      })
    } else {
      this.setData({ hasMore: false })
    }
  },

  // 商品筛选 - 实现搜索功能
  filterProducts(products) {
    if (!this.data.searchKeyword) return products
    
    const keyword = this.data.searchKeyword.toLowerCase()
    return products.filter(p => 
      p.name.toLowerCase().includes(keyword) || 
      p.id.toLowerCase().includes(keyword) ||
      p.basePrice.toString().includes(keyword) ||
      (p.description && p.description.toLowerCase().includes(keyword))
    )
  },

  // 搜索输入处理
  onSearchInput(e) {
    const keyword = e.detail.value.trim()
    this.setData({
      searchKeyword: keyword,
      searching: keyword.length > 0
    })
    
    // 使用防抖优化搜索性能
    clearTimeout(this.searchTimer)
    this.searchTimer = setTimeout(() => {
      this.performSearch()
    }, 300)
  },

  // 执行搜索
  performSearch() {
    if (this.data.searching) {
      const filtered = this.filterProducts(this.data.allProducts)
      this.setData({ 
        displayedProducts: filtered,
        hasMore: false
      })
    } else {
      this.setData({
        displayedProducts: this.data.allProducts.slice(0, this.data.pageSize),
        page: 1,
        hasMore: this.data.allProducts.length > this.data.pageSize
      })
    }
  },

  // 搜索确认
  onSearchConfirm(e) {
    const keyword = e.detail.value.trim()
    if (keyword) {
      this.setData({
        searchKeyword: keyword,
        searching: true
      })
      this.performSearch()
    }
  },

  // 清空搜索
  clearSearch() {
    this.setData({
      searchKeyword: '',
      searching: false,
      displayedProducts: this.data.allProducts.slice(0, this.data.pageSize),
      page: 1,
      hasMore: this.data.allProducts.length > this.data.pageSize
    })
  },

  // 滚动加载更多
  onReachBottom() {
    if (!this.data.searching && this.data.hasMore) {
      this.loadMoreProducts()
    }
  },

  // 选择商品
  selectProduct(e) {
    const product = e.currentTarget.dataset.product
    this.setData({ 
      selectedProduct: this.data.selectedProduct && this.data.selectedProduct.id === product.id ? null : product 
    })
  },
  // 重置/启动客户端超时定时器
  resetInactivityTimer() {
    // 先清除旧的定时器
    if (this.data.inactivityTimer) {
      clearTimeout(this.data.inactivityTimer);
    }
    
    // 设置一个新的定时器，例如2分钟（120000毫秒）
    const timer = setTimeout(() => {
      wx.showToast({
        title: '您长时间未操作，会话已自动结束',
        icon: 'none',
        duration: 2000
      });
      // 自动返回商品列表页
      this.backToProducts();
    }, 4000000); 

    this.setData({ inactivityTimer: timer });
  },

  // 开始谈判
  startNegotiation() {
    if (!this.data.userId) {
      return
    }
    if (!this.data.selectedProduct) {
      wx.showToast({ title: '请先选择商品', icon: 'none' });
      return
    }
    wx.showLoading({ title: '启动谈判中...' });
    wx.request({
      url: API_BASE + '/start_negotiation',
      method: 'POST',
      data: { 
        user_id: this.data.userId,
        product_id: this.data.selectedProduct.id
      },
      header: { 'Content-Type': 'application/json' },
      success: res => {
        wx.hideLoading(); // 显式隐藏加载提示
        if (res.statusCode === 200 && res.data.status === 'started') {
            // 清空旧消息，显示欢迎信息
          this.setData({ messages: [] });
          this.addMessage('system', res.data.welcome_message, false)// 添加欢迎消息
          // 更新状态
          this.setData({ 
            isNegotiating: true,
            negotiationStatus: 'ongoing',
            productInfo: res.data.product_info,
            welcomeMessage: res.data.welcome_message
          });
          // 【关键】谈判开始，启动客户端超时定时器
          this.resetInactivityTimer();
        } else {
          this.showError(`启动失败: ${res.data.error || res.statusCode}`);
        }
      },
      fail: err => {
        wx.hideLoading();
        this.showError(`网络错误: ${err.errMsg || '未知错误'}`);
      }
    });
    // 在发送请求之前添加延时（例如 2 秒）
    setTimeout(() => {
    // 执行请求
  }, 2000)  // 2000 毫秒即 2 秒延迟
  },

  onInput(e) {
    this.setData({ inputValue: e.detail.value })
  },

  // 发送用户消息
  sendMessage() {
      // 如果已经在等待机器人回复，则不执行
    const message = this.data.inputValue.trim()
    if (!message || this.data.isWaitingForBot) return;
    if (this.data.negotiationStatus !== 'ongoing') {
      wx.showToast({ title: '请先开始谈判', icon: 'none' })
      return
    }
    // 【关键】用户有操作，重置超时定时器
    this.resetInactivityTimer();
    // 1. 在界面上添加用户消息，并清空输入框
    this.addMessage('user', message, true)
    this.setData({ inputValue: ''})
    this.sendMessageInternal(message);
  },
  sendMessageInternal(message) {
    const TIMEOUT = 4000000; // 单次请求超时时间（100秒）
     // 2. 锁定输入，避免多次发送
     this.setData({ isWaitingForBot: true });
     wx.showLoading({ title: '等待回复中...' }); // 显示加载提示
      // 3. 发送请求并等待响应
    wx.request({
        url: API_BASE + '/chat',
        method: 'POST',
        data: { 
          message: message, user_id: this.data.userId
        },
        header: { 'Content-Type': 'application/json' },
        timeout: TIMEOUT, // 设置单次请求超时
        success: res => {
          if (res.statusCode === 200 && res.data.response) {
            // 4. 收到回复
            this.addMessage('agent', res.data.response); // 使用'agent' 作为角色标识
            // 判断是否谈判结束
            if (res.data.action === 'end') {
                this.handleNegotiationEnd(res.data.deal_price);
            }
          } else {
            // 处理服务器错误
            this.showError(`服务器错误: ${res.data.error || '未知错误'}`);
            this.handleNegotiationEnd(null);
          }
        },
        fail: err => {
          // 处理网络错误
          this.showError(`网络错误: ${err.errMsg || '未知'}`);
          this.handleNegotiationEnd(null); // 网络错误也结束
        },
        complete: () => {
          // 5. 不论成功与否，解锁输入
          this.setData({ isWaitingForBot: false });
          wx.hideLoading();
        }
      })
  },
    // 结束谈判
    endNegotiation() {
        if (this.data.negotiationStatus !== 'ongoing'  || this.data.isWaitingForBot) return
        const endMessage  = "抱歉，我们没法成交。"
        this.addMessage('user', endMessage , true)
        this.setData({ inputValue: '' })
         // 复用普通发送逻辑
         this.sendMessageInternal(endMessage);
      },
  // 处理谈判结束并显示结果
  handleNegotiationEnd(dealPrice) {
    // 【关键】谈判结束，清除定时器
    if (this.data.inactivityTimer) {
        clearTimeout(this.data.inactivityTimer);
        this.setData({ inactivityTimer: null });
      }
    this.setData({ negotiationStatus: 'ended' });
    let modalContent;
    if (dealPrice !== null && dealPrice !== undefined) {
        // 显示“交易价格”
        modalContent = `交易价格: ¥${dealPrice.toFixed(2)}`;
    } else {
        modalContent = '未达成交易';
    }
    wx.showModal({
    title: '谈判已结束',
    content: modalContent,
    showCancel: false,
    success: (res) => {
        if (res.confirm) {
          // 只有在用户确认后才重置谈判状态
          this.restartNegotiation();
        }
      }
    });
  },
  // 添加消息到聊天记录
  addMessage(role, content, isBuyer) {
    const newMessage = {
      id: this.data.messageId + 1,
      role: role,
      content: content,
      isBuyer: isBuyer || false,
      timestamp: new Date().toLocaleTimeString('en-US', { hour12: true })
    }
    
    const newMessages = this.data.messages.concat([newMessage])
    
    this.setData({
      messages: newMessages,
      messageId: newMessage.id,
      scrollTop: 10000 // 滚动到底部
    })
  },
  // 重新开始谈判
  restartNegotiation() {
    this.setData({
      messages: [],
      messageId: 0,
      negotiationStatus: 'not_started',
      selectedProduct: null,
      productInfo: null,
      isNegotiating: false,
      welcomeMessage: ''
    })
  },

  // 返回商品列表
  backToProducts() {
    // 【关键】返回时清除定时器
    if (this.data.inactivityTimer) {
        clearTimeout(this.data.inactivityTimer);
        this.setData({ inactivityTimer: null });
      }
    this.setData({
      isNegotiating: false,
      negotiationStatus: 'not_started',
      messages: [],
      messageId: 0,
      selectedProduct: null
    });
  },

  showError(msg) {
    wx.showToast({
      title: msg,
      icon: 'none',
      duration: 3000
    })
    this.addMessage('system', `错误: ${msg}`, false)
  },

  // 下拉刷新
  onPullDownRefresh() {
    this.setData({
      page: 1,
      hasMore: true,
      loading: false
    })
    this.loadProductsFromJson()
    wx.stopPullDownRefresh()
  }
})