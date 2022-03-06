#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


get_ipython().run_line_magic('config', "InlineBackend.figure.format = 'svg'")


# In[6]:


x = [1, 2, 3, 4, 5, 6, 7]
y = [3.5, 3.8, 4.2, 4.5, 5, 5.5, 7]


# In[7]:


plt.plot(x,y)
plt.show()


# In[8]:


plt.scatter(x, y, s= 60)
plt.show()


# In[9]:


t= np.linspace(0, 10, 51)


# In[10]:


f= np.cos(t)


# In[11]:


f


# In[13]:


plt.plot(t, f)


# In[24]:


plt.plot(t,f)
plt.title ('График f(t)', fontsize=16, fontweight='bold')
plt.xlabel('Значения t')
plt.ylabel('Значения f')
plt.axis([0.5, 9.5, -2.5, 2.5])


# In[25]:


x = np.linspace(-3, 3, 51)
print(x)


# In[26]:


y1 = x**2
print(y1)


# In[27]:


y2 = 2 * x + 0.5
print(y2)


# In[28]:


y3 = -3 * x - 1.5
print(y3)


# In[29]:


y4 = np.sin(x)
print(y4)


# In[30]:


fig, ax = plt.subplots(nrows=2, ncols=2)
ax1, ax2, ax3, ax4 = ax.flatten()
ax1.plot(x, y1)
ax2.plot(x, y2)
ax3.plot(x, y3)
ax4.plot(x, y4)
ax1.set_title('График $y_1$')
ax2.set_title('График $y_2$')
ax3.set_title('График $y_3$')
ax4.set_title('График $y_4$')
ax1.set_xlim([-5, 5])
fig.set_size_inches(8, 6)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()


# In[ ]:




