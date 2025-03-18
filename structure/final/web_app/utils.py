

def num_stats(train, test, columns):
    comparison = {}
    for col in columns:
        comparison[col] = {
            'DF Mean': train[col].mean(),
            'Test Mean': test[col].mean(),
            'DF Std': train[col].std(),
            'Test Std': test[col].std(),
            'DF Min': train[col].min(),
            'Test Min': test[col].min(),
            'DF 25%': train[col].quantile(0.25),
            'Test 25%': test[col].quantile(0.25),
            'DF 50%': train[col].median(),
            'Test 50%': test[col].median(),
            'DF 75%': train[col].quantile(0.75),
            'Test 75%': test[col].quantile(0.75),
            'DF Max': train[col].max(),
            'Test Max': test[col].max(),
        }
    return comparison

def obj_stats(train, test, columns):
    comparison = {}
    
    for col in columns:
        if col == 'Claim Injury Type':
            continue
        else:
            comparison[col] = {
                'DF Unique': train[col].nunique(),
                'Test Unique': test[col].nunique(),
                'DF Mode': train[col].mode()[0],
                'Test Mode': test[col].mode()[0],
                'DF Top Value Count': train[col].value_counts().iloc[0],
                'Test Top Value Count': test[col].value_counts().iloc[0],
        }
    return comparison


def group_industry(industry):
    # Public Services / Government
    if industry in ['PUBLIC ADMINISTRATION', 'HEALTH CARE AND SOCIAL ASSISTANCE', 'EDUCATIONAL SERVICES', 'ARTS, ENTERTAINMENT, AND RECREATION']:
        return 'Public Services / Government'
    
    # Business Services
    elif industry in ['PROFESSIONAL, SCIENTIFIC, AND TECHNICAL SERVICES', 'ADMINISTRATIVE AND SUPPORT AND WASTE MANAGEMENT AND REMEDIAT', 'INFORMATION', 
                      'MANAGEMENT OF COMPANIES AND ENTERPRISES', 'REAL ESTATE AND RENTAL AND LEASING', 'FINANCE AND INSURANCE']:
        return 'Business Services'
    
    # Retail and Wholesale
    elif industry in ['RETAIL TRADE', 'WHOLESALE TRADE', 'ACCOMMODATION AND FOOD SERVICES']:
        return 'Retail and Wholesale'
    
    # Manufacturing and Construction
    elif industry in ['MANUFACTURING', 'CONSTRUCTION']:
        return 'Manufacturing and Construction'
    
    # Transportation
    elif industry == 'TRANSPORTATION AND WAREHOUSING':
        return 'Transportation'
    
    # Agriculture and Natural Resources
    elif industry in ['AGRICULTURE, FORESTRY, FISHING AND HUNTING', 'MINING']:
        return 'Agriculture and Natural Resources'
    
    # Utilities
    elif industry == 'UTILITIES':
        return 'Utilities'
    
    # Return 'Other' if not in any of the categories
    else:
        return 'Other Services'
